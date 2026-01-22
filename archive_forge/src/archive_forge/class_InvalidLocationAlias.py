from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class InvalidLocationAlias(DirectoryLookupFailure):
    _fmt = '"%(alias_name)s" is not a valid location alias.'

    def __init__(self, alias_name):
        DirectoryLookupFailure.__init__(self, alias_name=alias_name)