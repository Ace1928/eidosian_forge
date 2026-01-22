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
def _get_filtered_chk_streams(self, excluded_revision_keys):
    self._text_keys = set()
    excluded_revision_keys.discard(_mod_revision.NULL_REVISION)
    if not excluded_revision_keys:
        uninteresting_root_keys = set()
        uninteresting_pid_root_keys = set()
    else:
        present_keys = self.from_repository._find_present_inventory_keys(excluded_revision_keys)
        present_ids = [k[-1] for k in present_keys]
        uninteresting_root_keys = set()
        uninteresting_pid_root_keys = set()
        for inv in self.from_repository.iter_inventories(present_ids):
            uninteresting_root_keys.add(inv.id_to_entry.key())
            uninteresting_pid_root_keys.add(inv.parent_id_basename_to_file_id.key())
    chk_bytes = self.from_repository.chk_bytes

    def _filter_id_to_entry():
        interesting_nodes = chk_map.iter_interesting_nodes(chk_bytes, self._chk_id_roots, uninteresting_root_keys)
        for record in _filter_text_keys(interesting_nodes, self._text_keys, chk_map._bytes_to_text_key):
            if record is not None:
                yield record
        self._chk_id_roots = None
    yield ('chk_bytes', _filter_id_to_entry())

    def _get_parent_id_basename_to_file_id_pages():
        for record, items in chk_map.iter_interesting_nodes(chk_bytes, self._chk_p_id_roots, uninteresting_pid_root_keys):
            if record is not None:
                yield record
        self._chk_p_id_roots = None
    yield ('chk_bytes', _get_parent_id_basename_to_file_id_pages())