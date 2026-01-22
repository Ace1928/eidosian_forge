from __future__ import annotations
import re
import sys
import typing
from collections.abc import MutableMapping, Sequence
from urwid import str_util
import urwid.util  # isort: skip  # pylint: disable=wrong-import-position
class KeyqueueTrie:
    __slots__ = ('data',)

    def __init__(self, sequences: Iterable[tuple[str, str]]) -> None:
        self.data: dict[int, str | dict[int, str | dict[int, str]]] = {}
        for s, result in sequences:
            if isinstance(result, dict):
                raise TypeError(result)
            self.add(self.data, s, result)

    def add(self, root: MutableMapping[int, str | MutableMapping[int, str | MutableMapping[int, str]]], s: str, result: str) -> None:
        if not isinstance(root, MutableMapping) or not s:
            raise RuntimeError('trie conflict detected')
        if ord(s[0]) in root:
            self.add(root[ord(s[0])], s[1:], result)
            return
        if len(s) > 1:
            d = {}
            root[ord(s[0])] = d
            self.add(d, s[1:], result)
            return
        root[ord(s)] = result

    def get(self, keys, more_available: bool):
        result = self.get_recurse(self.data, keys, more_available)
        if not result:
            result = self.read_cursor_position(keys, more_available)
        return result

    def get_recurse(self, root: MutableMapping[int, str | MutableMapping[int, str | MutableMapping[int, str]]] | typing.Literal['mouse', 'sgrmouse'], keys: Collection[int], more_available: bool):
        if not isinstance(root, MutableMapping):
            if root == 'mouse':
                return self.read_mouse_info(keys, more_available)
            if root == 'sgrmouse':
                return self.read_sgrmouse_info(keys, more_available)
            return (root, keys)
        if not keys:
            if more_available:
                raise MoreInputRequired()
            return None
        if keys[0] not in root:
            return None
        return self.get_recurse(root[keys[0]], keys[1:], more_available)

    def read_mouse_info(self, keys: Collection[int], more_available: bool):
        if len(keys) < 3:
            if more_available:
                raise MoreInputRequired()
            return None
        b = keys[0] - 32
        x, y = ((keys[1] - 33) % 256, (keys[2] - 33) % 256)
        prefixes = []
        if b & 4:
            prefixes.append('shift ')
        if b & 8:
            prefixes.append('meta ')
        if b & 16:
            prefixes.append('ctrl ')
        if (b & MOUSE_MULTIPLE_CLICK_MASK) >> 9 == 1:
            prefixes.append('double ')
        if (b & MOUSE_MULTIPLE_CLICK_MASK) >> 9 == 2:
            prefixes.append('triple ')
        prefix = ''.join(prefixes)
        button = (b & 64) // 64 * 3 + (b & 3) + 1
        if b & 3 == 3:
            action = 'release'
            button = 0
        elif b & MOUSE_RELEASE_FLAG:
            action = 'release'
        elif b & MOUSE_DRAG_FLAG:
            action = 'drag'
        elif b & MOUSE_MULTIPLE_CLICK_MASK:
            action = 'click'
        else:
            action = 'press'
        return ((f'{prefix}mouse {action}', button, x, y), keys[3:])

    def read_sgrmouse_info(self, keys: Collection[int], more_available: bool):
        if not keys:
            if more_available:
                raise MoreInputRequired()
            return None
        value = ''
        pos_m = 0
        found_m = False
        for k in keys:
            value += chr(k)
            if k in {ord('M'), ord('m')}:
                found_m = True
                break
            pos_m += 1
        if not found_m:
            if more_available:
                raise MoreInputRequired()
            return None
        b, x, y = (int(val) for val in value[:-1].split(';'))
        action = value[-1]
        prefixes = []
        if b & 4:
            prefixes.append('shift ')
        if b & 8:
            prefixes.append('meta ')
        if b & 16:
            prefixes.append('ctrl ')
        prefix = ''.join(prefixes)
        wheel_used: typing.Literal[0, 1] = (b & 64) >> 6
        button = wheel_used * 3 + (b & 3) + 1
        x -= 1
        y -= 1
        if action == 'M':
            if b & MOUSE_DRAG_FLAG:
                action = 'drag'
            else:
                action = 'press'
        elif action == 'm':
            action = 'release'
        else:
            raise ValueError(f'Unknown mouse action: {action!r}')
        return ((f'{prefix}mouse {action}', button, x, y), keys[pos_m + 1:])

    def read_cursor_position(self, keys, more_available: bool):
        """
        Interpret cursor position information being sent by the
        user's terminal.  Returned as ('cursor position', x, y)
        where (x, y) == (0, 0) is the top left of the screen.
        """
        if not keys:
            if more_available:
                raise MoreInputRequired()
            return None
        if keys[0] != ord('['):
            return None
        y = 0
        i = 1
        for k in keys[i:]:
            i += 1
            if k == ord(';'):
                if not y:
                    return None
                break
            if k < ord('0') or k > ord('9'):
                return None
            if not y and k == ord('0'):
                return None
            y = y * 10 + k - ord('0')
        if not keys[i:]:
            if more_available:
                raise MoreInputRequired()
            return None
        x = 0
        for k in keys[i:]:
            i += 1
            if k == ord('R'):
                if not x:
                    return None
                return (('cursor position', x - 1, y - 1), keys[i:])
            if k < ord('0') or k > ord('9'):
                return None
            if not x and k == ord('0'):
                return None
            x = x * 10 + k - ord('0')
        if not keys[i:] and more_available:
            raise MoreInputRequired()
        return None