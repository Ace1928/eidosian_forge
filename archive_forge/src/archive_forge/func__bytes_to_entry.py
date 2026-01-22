from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def _bytes_to_entry(self, bytes):
    """Deserialise a serialised entry."""
    sections = bytes.split(b'\n')
    if sections[0].startswith(b'file: '):
        result = InventoryFile(sections[0][6:], sections[2].decode('utf8'), sections[1])
        result.text_sha1 = sections[4]
        result.text_size = int(sections[5])
        result.executable = sections[6] == b'Y'
    elif sections[0].startswith(b'dir: '):
        result = CHKInventoryDirectory(sections[0][5:], sections[2].decode('utf8'), sections[1], self)
    elif sections[0].startswith(b'symlink: '):
        result = InventoryLink(sections[0][9:], sections[2].decode('utf8'), sections[1])
        result.symlink_target = sections[4].decode('utf8')
    elif sections[0].startswith(b'tree: '):
        result = TreeReference(sections[0][6:], sections[2].decode('utf8'), sections[1])
        result.reference_revision = sections[4]
    else:
        raise ValueError('Not a serialised entry %r' % bytes)
    result.file_id = result.file_id
    result.revision = sections[3]
    if result.parent_id == b'':
        result.parent_id = None
    self._fileid_to_entry_cache[result.file_id] = result
    return result