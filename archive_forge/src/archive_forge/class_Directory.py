from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class Directory:
    """Abstract directory lookup class."""

    def look_up(self, name, url, purpose=None):
        """Look up an entry in a directory.

        :param name: Directory name
        :param url: The URL to dereference
        :param purpose: Purpose of the URL ('read', 'write' or None - if not declared)
        :return: The dereferenced URL if applicable, the input URL otherwise.
        """
        raise NotImplementedError(self.look_up)