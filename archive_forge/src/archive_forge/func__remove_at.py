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
def _remove_at(self, idx: int) -> None:
    key = self._body[idx][0]
    index = self._map.get(key)
    if index is None:
        raise NonExistentKey(key)
    self._body[idx] = (None, Null())
    if isinstance(index, tuple):
        index = list(index)
        index.remove(idx)
        if len(index) == 1:
            index = index.pop()
        else:
            index = tuple(index)
        self._map[key] = index
    else:
        dict.__delitem__(self, key.key)
        self._map.pop(key)