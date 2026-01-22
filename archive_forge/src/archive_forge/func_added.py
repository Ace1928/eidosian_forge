import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
def added(kind, extra, lines):
    info = extra.split(' // ')
    if len(info) <= 1:
        raise BzrError('add action lines require the path and file id: %r' % extra)
    elif len(info) > 5:
        raise BzrError('add action lines have fewer than 5 entries.: %r' % extra)
    path = info[0]
    if not info[1].startswith('file-id:'):
        raise BzrError('The file-id should follow the path for an add: %r' % extra)
    file_id = cache_utf8.encode(info[1][8:])
    bundle_tree.note_id(file_id, path, kind)
    bundle_tree.note_executable(path, False)
    last_changed, encoding = extra_info(info[2:], path)
    revision = get_rev_id(last_changed, path, kind)
    if kind == 'directory':
        return
    do_patch(path, lines, encoding)