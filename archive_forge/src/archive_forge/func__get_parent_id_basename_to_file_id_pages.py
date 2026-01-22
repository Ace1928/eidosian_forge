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
def _get_parent_id_basename_to_file_id_pages():
    for record, items in chk_map.iter_interesting_nodes(chk_bytes, self._chk_p_id_roots, uninteresting_pid_root_keys):
        if record is not None:
            yield record
    self._chk_p_id_roots = None