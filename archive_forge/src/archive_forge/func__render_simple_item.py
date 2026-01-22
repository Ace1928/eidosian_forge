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
def _render_simple_item(self, key, item, prefix=None):
    if key is None:
        return item.as_string()
    _key = key.as_string()
    if prefix is not None:
        _key = prefix + '.' + _key
    return f'{item.trivia.indent}{decode(_key)}{key.sep}{decode(item.as_string())}{item.trivia.comment_ws}{decode(item.trivia.comment)}{item.trivia.trail}'