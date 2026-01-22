from __future__ import annotations
import enum
import typing
from typing import Iterator
class CommandMap(typing.Mapping[str, typing.Union[str, Command, None]]):
    """
    dict-like object for looking up commands from keystrokes

    Default values (key: command)::

        'tab':       'next selectable',
        'ctrl n':    'next selectable',
        'shift tab': 'prev selectable',
        'ctrl p':    'prev selectable',
        'ctrl l':    'redraw screen',
        'esc':       'menu',
        'up':        'cursor up',
        'down':      'cursor down',
        'left':      'cursor left',
        'right':     'cursor right',
        'page up':   'cursor page up',
        'page down': 'cursor page down',
        'home':      'cursor max left',
        'end':       'cursor max right',
        ' ':         'activate',
        'enter':     'activate',
    """

    def __iter__(self) -> Iterator[str]:
        return iter(self._command)

    def __len__(self) -> int:
        return len(self._command)
    _command_defaults: typing.ClassVar[dict[str, str | Command]] = {'tab': Command.SELECT_NEXT, 'ctrl n': Command.SELECT_NEXT, 'shift tab': Command.SELECT_PREVIOUS, 'ctrl p': Command.SELECT_PREVIOUS, 'ctrl l': Command.REDRAW_SCREEN, 'esc': Command.MENU, 'up': Command.UP, 'down': Command.DOWN, 'left': Command.LEFT, 'right': Command.RIGHT, 'page up': Command.PAGE_UP, 'page down': Command.PAGE_DOWN, 'home': Command.MAX_LEFT, 'end': Command.MAX_RIGHT, ' ': Command.ACTIVATE, 'enter': Command.ACTIVATE}

    def __init__(self) -> None:
        self._command = dict(self._command_defaults)

    def restore_defaults(self) -> None:
        self._command = dict(self._command_defaults)

    def __getitem__(self, key: str) -> str | Command | None:
        return self._command.get(key, None)

    def __setitem__(self, key, command: str | Command) -> None:
        self._command[key] = command

    def __delitem__(self, key: str) -> None:
        del self._command[key]

    def clear_command(self, command: str | Command) -> None:
        dk = [k for k, v in self._command.items() if v == command]
        for k in dk:
            del self._command[k]

    def copy(self) -> Self:
        """
        Return a new copy of this CommandMap, likely so we can modify
        it separate from a shared one.
        """
        c = self.__class__()
        c._command = dict(self._command)
        return c