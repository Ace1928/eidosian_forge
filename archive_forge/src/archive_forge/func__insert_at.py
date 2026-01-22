from __future__ import annotations
import copy
from typing import Any
from typing import Iterator
from tomlkit._compat import decode
from tomlkit._types import _CustomDict
from tomlkit._utils import merge_dicts
from tomlkit.exceptions import KeyAlreadyPresent
from tomlkit.exceptions import NonExistentKey
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import AoT
from tomlkit.items import Comment
from tomlkit.items import Item
from tomlkit.items import Key
from tomlkit.items import Null
from tomlkit.items import SingleKey
from tomlkit.items import Table
from tomlkit.items import Trivia
from tomlkit.items import Whitespace
from tomlkit.items import item as _item
def _insert_at(self, idx: int, key: Key | str, item: Any) -> Container:
    if idx > len(self._body) - 1:
        raise ValueError(f'Unable to insert at position {idx}')
    if not isinstance(key, Key):
        key = SingleKey(key)
    item = _item(item)
    if idx > 0:
        previous_item = self._body[idx - 1][1]
        if not (isinstance(previous_item, Whitespace) or ends_with_whitespace(previous_item) or isinstance(item, (AoT, Table)) or ('\n' in previous_item.trivia.trail)):
            previous_item.trivia.trail += '\n'
    for k, v in self._map.items():
        if isinstance(v, tuple):
            new_indices = []
            for v_ in v:
                if v_ >= idx:
                    v_ = v_ + 1
                new_indices.append(v_)
            self._map[k] = tuple(new_indices)
        elif v >= idx:
            self._map[k] = v + 1
    if key in self._map:
        current_idx = self._map[key]
        if not isinstance(current_idx, tuple):
            current_idx = (current_idx,)
        self._map[key] = current_idx + (idx,)
    else:
        self._map[key] = idx
    self._body.insert(idx, (key, item))
    dict.__setitem__(self, key.key, item.value)
    return self