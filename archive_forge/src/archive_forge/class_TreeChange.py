import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
class TreeChange(namedtuple('TreeChange', ['type', 'old', 'new'])):
    """Named tuple a single change between two trees."""

    @classmethod
    def add(cls, new):
        return cls(CHANGE_ADD, _NULL_ENTRY, new)

    @classmethod
    def delete(cls, old):
        return cls(CHANGE_DELETE, old, _NULL_ENTRY)