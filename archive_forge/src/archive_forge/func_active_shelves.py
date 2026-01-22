import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def active_shelves(self):
    """Return a list of shelved changes."""
    active = sorted(self.get_shelf_ids(self.transport.list_dir('.')))
    return active