from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
class TreeLink(TreeEntry):
    """See TreeEntry. This is a symlink in a working tree."""
    __slots__: List[str] = []
    kind = 'symlink'

    def kind_character(self):
        return ''