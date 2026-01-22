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
def add_sha(d, revision_id, sha1):
    if revision_id is None:
        if sha1 is not None:
            raise BzrError('A Null revision should alwayshave a null sha1 hash')
        return
    if revision_id in d:
        if sha1 != d[revision_id]:
            raise BzrError('** Revision %r referenced with 2 different sha hashes %s != %s' % (revision_id, sha1, d[revision_id]))
    else:
        d[revision_id] = sha1