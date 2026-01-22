from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
@staticmethod
def _bytes_to_utf8name_key(data):
    """Get the file_id, revision_id key out of data."""
    sections = data.split(b'\n')
    kind, file_id = sections[0].split(b': ')
    return (sections[2], file_id, sections[3])