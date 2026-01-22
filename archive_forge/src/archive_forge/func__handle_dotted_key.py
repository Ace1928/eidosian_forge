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
def _handle_dotted_key(self, key: Key, value: Item) -> None:
    if isinstance(value, (Table, AoT)):
        raise TOMLKitError("Can't add a table to a dotted key")
    name, *mid, last = key
    name._dotted = True
    table = current = Table(Container(True), Trivia(), False, is_super_table=True)
    for _name in mid:
        _name._dotted = True
        new_table = Table(Container(True), Trivia(), False, is_super_table=True)
        current.append(_name, new_table)
        current = new_table
    last.sep = key.sep
    current.append(last, value)
    self.append(name, table)
    return