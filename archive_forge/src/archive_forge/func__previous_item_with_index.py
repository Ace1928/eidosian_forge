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
def _previous_item_with_index(self, idx: int | None=None, ignore=(Null,)) -> tuple[int, Item] | None:
    """Find the immediate previous item before index ``idx``"""
    if idx is None or idx > len(self._body):
        idx = len(self._body)
    for i in range(idx - 1, -1, -1):
        v = self._body[i][-1]
        if not isinstance(v, ignore):
            return (i, v)
    return None