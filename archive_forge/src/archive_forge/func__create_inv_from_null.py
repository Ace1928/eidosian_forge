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
def _create_inv_from_null(self, delta, revision_id):
    """This will mutate new_inv directly.

        This is a simplified form of create_by_apply_delta which knows that all
        the old values must be None, so everything is a create.
        """
    serializer = self._format._serializer
    new_inv = inventory.CHKInventory(serializer.search_key_name)
    new_inv.revision_id = revision_id
    entry_to_bytes = new_inv._entry_to_bytes
    id_to_entry_dict = {}
    parent_id_basename_dict = {}
    for old_path, new_path, file_id, entry in delta:
        if old_path is not None:
            raise ValueError('Invalid delta, somebody tried to delete %r from the NULL_REVISION' % ((old_path, file_id),))
        if new_path is None:
            raise ValueError('Invalid delta, delta from NULL_REVISION has no new_path %r' % (file_id,))
        if new_path == '':
            new_inv.root_id = file_id
            parent_id_basename_key = StaticTuple(b'', b'').intern()
        else:
            utf8_entry_name = entry.name.encode('utf-8')
            parent_id_basename_key = StaticTuple(entry.parent_id, utf8_entry_name).intern()
        new_value = entry_to_bytes(entry)
        key = StaticTuple(file_id).intern()
        id_to_entry_dict[key] = new_value
        parent_id_basename_dict[parent_id_basename_key] = file_id
    new_inv._populate_from_dicts(self.chk_bytes, id_to_entry_dict, parent_id_basename_dict, maximum_size=serializer.maximum_size)
    return new_inv