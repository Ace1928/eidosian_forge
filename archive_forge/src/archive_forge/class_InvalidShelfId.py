import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
class InvalidShelfId(errors.BzrError):
    _fmt = '"%(invalid_id)s" is not a valid shelf id, try a number instead.'

    def __init__(self, invalid_id):
        errors.BzrError.__init__(self, invalid_id=invalid_id)