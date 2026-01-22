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
def _get_filtered_canonicalizing_inv_stream(self, source_vf, keys, message, pb=None, source_chk_vf=None, target_chk_vf=None):
    """Filter the texts of inventories, regenerating CHKs to make sure they
        are canonical.
        """
    total_keys = len(keys)
    target_chk_vf = versionedfile.NoDupeAddLinesDecorator(target_chk_vf)

    def _filtered_inv_stream():
        stream = source_vf.get_record_stream(keys, 'groupcompress', True)
        search_key_name = None
        for idx, record in enumerate(stream):
            lines = record.get_bytes_as('lines')
            chk_inv = inventory.CHKInventory.deserialise(source_chk_vf, lines, record.key)
            if pb is not None:
                pb.update('inv', idx, total_keys)
            chk_inv.id_to_entry._ensure_root()
            if search_key_name is None:
                search_key_reg = chk_map.search_key_registry
                for search_key_name, func in search_key_reg.items():
                    if func == chk_inv.id_to_entry._search_key_func:
                        break
            canonical_inv = inventory.CHKInventory.from_inventory(target_chk_vf, chk_inv, maximum_size=chk_inv.id_to_entry._root_node._maximum_size, search_key_name=search_key_name)
            if chk_inv.id_to_entry.key() != canonical_inv.id_to_entry.key():
                trace.mutter('Non-canonical CHK map for id_to_entry of inv: %s (root is %s, should be %s)' % (chk_inv.revision_id, chk_inv.id_to_entry.key()[0], canonical_inv.id_to_entry.key()[0]))
                self._data_changed = True
            p_id_map = chk_inv.parent_id_basename_to_file_id
            p_id_map._ensure_root()
            canon_p_id_map = canonical_inv.parent_id_basename_to_file_id
            if p_id_map.key() != canon_p_id_map.key():
                trace.mutter('Non-canonical CHK map for parent_id_to_basename of inv: %s (root is %s, should be %s)' % (chk_inv.revision_id, p_id_map.key()[0], canon_p_id_map.key()[0]))
                self._data_changed = True
            yield versionedfile.ChunkedContentFactory(record.key, record.parents, record.sha1, canonical_inv.to_lines(), chunks_are_lines=True)
    return _filtered_inv_stream()