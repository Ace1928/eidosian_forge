import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class IllegalMergeDirectivePayload(errors.BzrError):
    """A merge directive contained something other than a patch or bundle"""
    _fmt = 'Bad merge directive payload %(start)r'

    def __init__(self, start):
        errors.BzrError(self)
        self.start = start