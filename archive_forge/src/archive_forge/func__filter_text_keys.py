import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def _filter_text_keys(interesting_nodes_iterable, text_keys, bytes_to_text_key):
    """Iterate the result of iter_interesting_nodes, yielding the records
    and adding to text_keys.
    """
    text_keys_update = text_keys.update
    for record, items in interesting_nodes_iterable:
        text_keys_update([bytes_to_text_key(b) for n, b in items])
        yield record