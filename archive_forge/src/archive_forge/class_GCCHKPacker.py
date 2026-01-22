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
class GCCHKPacker(Packer):
    """This class understand what it takes to collect a GCCHK repo."""

    def __init__(self, pack_collection, packs, suffix, revision_ids=None, reload_func=None):
        super().__init__(pack_collection, packs, suffix, revision_ids=revision_ids, reload_func=reload_func)
        self._pack_collection = pack_collection
        if pack_collection.chk_index is None:
            raise AssertionError('pack_collection.chk_index should not be None')
        self._gather_text_refs = False
        self._chk_id_roots = []
        self._chk_p_id_roots = []
        self._text_refs = None
        self.revision_keys = None

    def _get_progress_stream(self, source_vf, keys, message, pb):

        def pb_stream():
            substream = source_vf.get_record_stream(keys, 'groupcompress', True)
            for idx, record in enumerate(substream):
                if pb is not None:
                    pb.update(message, idx + 1, len(keys))
                yield record
        return pb_stream()

    def _get_filtered_inv_stream(self, source_vf, keys, message, pb=None):
        """Filter the texts of inventories, to find the chk pages."""
        total_keys = len(keys)

        def _filtered_inv_stream():
            id_roots_set = set()
            p_id_roots_set = set()
            stream = source_vf.get_record_stream(keys, 'groupcompress', True)
            for idx, record in enumerate(stream):
                lines = record.get_bytes_as('lines')
                chk_inv = inventory.CHKInventory.deserialise(None, lines, record.key)
                if pb is not None:
                    pb.update('inv', idx, total_keys)
                key = chk_inv.id_to_entry.key()
                if key not in id_roots_set:
                    self._chk_id_roots.append(key)
                    id_roots_set.add(key)
                p_id_map = chk_inv.parent_id_basename_to_file_id
                if p_id_map is None:
                    raise AssertionError('Parent id -> file_id map not set')
                key = p_id_map.key()
                if key not in p_id_roots_set:
                    p_id_roots_set.add(key)
                    self._chk_p_id_roots.append(key)
                yield record
            id_roots_set.clear()
            p_id_roots_set.clear()
        return _filtered_inv_stream()

    def _get_chk_streams(self, source_vf, keys, pb=None):
        total_keys = len(keys)
        remaining_keys = set(keys)
        counter = [0]
        if self._gather_text_refs:
            self._text_refs = set()

        def _get_referenced_stream(root_keys, parse_leaf_nodes=False):
            cur_keys = root_keys
            while cur_keys:
                keys_by_search_prefix = {}
                remaining_keys.difference_update(cur_keys)
                next_keys = set()

                def handle_internal_node(node):
                    for prefix, value in node._items.items():
                        if value not in next_keys and value in remaining_keys:
                            keys_by_search_prefix.setdefault(prefix, []).append(value)
                            next_keys.add(value)

                def handle_leaf_node(node):
                    for file_id, bytes in node.iteritems(None):
                        self._text_refs.add(chk_map._bytes_to_text_key(bytes))

                def next_stream():
                    stream = source_vf.get_record_stream(cur_keys, 'as-requested', True)
                    for record in stream:
                        if record.storage_kind == 'absent':
                            continue
                        bytes = record.get_bytes_as('fulltext')
                        node = chk_map._deserialise(bytes, record.key, search_key_func=None)
                        common_base = node._search_prefix
                        if isinstance(node, chk_map.InternalNode):
                            handle_internal_node(node)
                        elif parse_leaf_nodes:
                            handle_leaf_node(node)
                        counter[0] += 1
                        if pb is not None:
                            pb.update('chk node', counter[0], total_keys)
                        yield record
                yield next_stream()
                cur_keys = []
                for prefix in sorted(keys_by_search_prefix):
                    cur_keys.extend(keys_by_search_prefix.pop(prefix))
        for stream in _get_referenced_stream(self._chk_id_roots, self._gather_text_refs):
            yield stream
        del self._chk_id_roots
        chk_p_id_roots = [key for key in self._chk_p_id_roots if key in remaining_keys]
        del self._chk_p_id_roots
        for stream in _get_referenced_stream(chk_p_id_roots, False):
            yield stream
        if remaining_keys:
            trace.mutter('There were %d keys in the chk index, %d of which were not referenced', total_keys, len(remaining_keys))
            if self.revision_ids is None:
                stream = source_vf.get_record_stream(remaining_keys, 'unordered', True)
                yield stream

    def _build_vf(self, index_name, parents, delta, for_write=False):
        """Build a VersionedFiles instance on top of this group of packs."""
        index_name = index_name + '_index'
        index_to_pack = {}
        access = _DirectPackAccess(index_to_pack, reload_func=self._reload_func)
        if for_write:
            if self.new_pack is None:
                raise AssertionError('No new pack has been set')
            index = getattr(self.new_pack, index_name)
            index_to_pack[index] = self.new_pack.access_tuple()
            index.set_optimize(for_size=True)
            access.set_writer(self.new_pack._writer, index, self.new_pack.access_tuple())
            add_callback = index.add_nodes
        else:
            indices = []
            for pack in self.packs:
                sub_index = getattr(pack, index_name)
                index_to_pack[sub_index] = pack.access_tuple()
                indices.append(sub_index)
            index = _mod_index.CombinedGraphIndex(indices)
            add_callback = None
        vf = GroupCompressVersionedFiles(_GCGraphIndex(index, add_callback=add_callback, parents=parents, is_locked=self._pack_collection.repo.is_locked), access=access, delta=delta)
        return vf

    def _build_vfs(self, index_name, parents, delta):
        """Build the source and target VersionedFiles."""
        source_vf = self._build_vf(index_name, parents, delta, for_write=False)
        target_vf = self._build_vf(index_name, parents, delta, for_write=True)
        return (source_vf, target_vf)

    def _copy_stream(self, source_vf, target_vf, keys, message, vf_to_stream, pb_offset):
        trace.mutter('repacking %d %s', len(keys), message)
        self.pb.update('repacking {}'.format(message), pb_offset)
        with ui.ui_factory.nested_progress_bar() as child_pb:
            stream = vf_to_stream(source_vf, keys, message, child_pb)
            for _, _ in target_vf._insert_record_stream(stream, random_id=True, reuse_blocks=False):
                pass

    def _copy_revision_texts(self):
        source_vf, target_vf = self._build_vfs('revision', True, False)
        if not self.revision_keys:
            self.revision_keys = source_vf.keys()
        self._copy_stream(source_vf, target_vf, self.revision_keys, 'revisions', self._get_progress_stream, 1)

    def _copy_inventory_texts(self):
        source_vf, target_vf = self._build_vfs('inventory', True, True)
        inventory_keys = source_vf.keys()
        missing_inventories = set(self.revision_keys).difference(inventory_keys)
        if missing_inventories:
            inv_index = self._pack_collection.repo.inventories._index
            pmap = inv_index.get_parent_map(missing_inventories)
            really_missing = missing_inventories.difference(pmap)
            if really_missing:
                missing_inventories = sorted(really_missing)
                raise ValueError('We are missing inventories for revisions: %s' % (missing_inventories,))
        self._copy_stream(source_vf, target_vf, inventory_keys, 'inventories', self._get_filtered_inv_stream, 2)

    def _get_chk_vfs_for_copy(self):
        return self._build_vfs('chk', False, False)

    def _copy_chk_texts(self):
        source_vf, target_vf = self._get_chk_vfs_for_copy()
        total_keys = source_vf.keys()
        trace.mutter('repacking chk: %d id_to_entry roots, %d p_id_map roots, %d total keys', len(self._chk_id_roots), len(self._chk_p_id_roots), len(total_keys))
        self.pb.update('repacking chk', 3)
        with ui.ui_factory.nested_progress_bar() as child_pb:
            for stream in self._get_chk_streams(source_vf, total_keys, pb=child_pb):
                for _, _ in target_vf._insert_record_stream(stream, random_id=True, reuse_blocks=False):
                    pass

    def _copy_text_texts(self):
        source_vf, target_vf = self._build_vfs('text', True, True)
        text_keys = source_vf.keys()
        self._copy_stream(source_vf, target_vf, text_keys, 'texts', self._get_progress_stream, 4)

    def _copy_signature_texts(self):
        source_vf, target_vf = self._build_vfs('signature', False, False)
        signature_keys = source_vf.keys()
        signature_keys.intersection(self.revision_keys)
        self._copy_stream(source_vf, target_vf, signature_keys, 'signatures', self._get_progress_stream, 5)

    def _create_pack_from_packs(self):
        self.pb.update('repacking', 0, 7)
        self.new_pack = self.open_pack()
        self.new_pack.set_write_cache_size(1024 * 1024)
        self._copy_revision_texts()
        self._copy_inventory_texts()
        self._copy_chk_texts()
        self._copy_text_texts()
        self._copy_signature_texts()
        self.new_pack._check_references()
        if not self._use_pack(self.new_pack):
            self.new_pack.abort()
            return None
        self.new_pack.finish_content()
        if len(self.packs) == 1:
            old_pack = self.packs[0]
            if old_pack.name == self.new_pack._hash.hexdigest():
                trace.mutter('single pack %s was already optimally packed', old_pack.name)
                self.new_pack.abort()
                return None
        self.pb.update('finishing repack', 6, 7)
        self.new_pack.finish()
        self._pack_collection.allocate(self.new_pack)
        return self.new_pack