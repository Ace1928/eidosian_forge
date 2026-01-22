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
class GCCHKReconcilePacker(GCCHKPacker):
    """A packer which regenerates indices etc as it copies.

    This is used by ``brz reconcile`` to cause parent text pointers to be
    regenerated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_changed = False
        self._gather_text_refs = True

    def _copy_inventory_texts(self):
        source_vf, target_vf = self._build_vfs('inventory', True, True)
        self._copy_stream(source_vf, target_vf, self.revision_keys, 'inventories', self._get_filtered_inv_stream, 2)
        if source_vf.keys() != self.revision_keys:
            self._data_changed = True

    def _copy_text_texts(self):
        """generate what texts we should have and then copy."""
        source_vf, target_vf = self._build_vfs('text', True, True)
        trace.mutter('repacking %d texts', len(self._text_refs))
        self.pb.update('repacking texts', 4)
        repo = self._pack_collection.repo
        revision_vf = self._build_vf('revision', True, False, for_write=True)
        ancestor_keys = revision_vf.get_parent_map(revision_vf.keys())
        ancestors = {k[0]: tuple([p[0] for p in parents]) for k, parents in ancestor_keys.items()}
        del ancestor_keys
        ideal_index = repo._generate_text_key_index(None, ancestors)
        file_id_parent_map = source_vf.get_parent_map(self._text_refs)
        ok_keys = []
        new_parent_keys = {}
        discarded_keys = []
        NULL_REVISION = _mod_revision.NULL_REVISION
        for key in self._text_refs:
            try:
                ideal_parents = tuple(ideal_index[key])
            except KeyError:
                discarded_keys.append(key)
                self._data_changed = True
            else:
                if ideal_parents == (NULL_REVISION,):
                    ideal_parents = ()
                source_parents = file_id_parent_map[key]
                if ideal_parents == source_parents:
                    ok_keys.append(key)
                else:
                    self._data_changed = True
                    new_parent_keys[key] = ideal_parents
        del ideal_index
        del file_id_parent_map

        def _update_parents_for_texts():
            stream = source_vf.get_record_stream(self._text_refs, 'groupcompress', False)
            for record in stream:
                if record.key in new_parent_keys:
                    record.parents = new_parent_keys[record.key]
                yield record
        target_vf.insert_record_stream(_update_parents_for_texts())

    def _use_pack(self, new_pack):
        """Override _use_pack to check for reconcile having changed content."""
        return new_pack.data_inserted() and self._data_changed