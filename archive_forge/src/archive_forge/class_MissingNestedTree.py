from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
class MissingNestedTree(errors.BzrError):
    _fmt = 'The nested tree for %(path)s can not be resolved.'

    def __init__(self, path):
        self.path = path