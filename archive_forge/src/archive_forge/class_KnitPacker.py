from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
class KnitPacker(Packer):
    """Packer that works with knit packs."""

    def __init__(self, pack_collection, packs, suffix, revision_ids=None, reload_func=None):
        super().__init__(pack_collection, packs, suffix, revision_ids=revision_ids, reload_func=reload_func)

    def _pack_map_and_index_list(self, index_attribute):
        """Convert a list of packs to an index pack map and index list.

        :param index_attribute: The attribute that the desired index is found
            on.
        :return: A tuple (map, list) where map contains the dict from
            index:pack_tuple, and list contains the indices in the preferred
            access order.
        """
        indices = []
        pack_map = {}
        for pack_obj in self.packs:
            index = getattr(pack_obj, index_attribute)
            indices.append(index)
            pack_map[index] = pack_obj
        return (pack_map, indices)

    def _index_contents(self, indices, key_filter=None):
        """Get an iterable of the index contents from a pack_map.

        :param indices: The list of indices to query
        :param key_filter: An optional filter to limit the keys returned.
        """
        all_index = CombinedGraphIndex(indices)
        if key_filter is None:
            return all_index.iter_all_entries()
        else:
            return all_index.iter_entries(key_filter)

    def _copy_nodes(self, nodes, index_map, writer, write_index, output_lines=None):
        """Copy knit nodes between packs with no graph references.

        :param output_lines: Output full texts of copied items.
        """
        with ui.ui_factory.nested_progress_bar() as pb:
            return self._do_copy_nodes(nodes, index_map, writer, write_index, pb, output_lines=output_lines)

    def _do_copy_nodes(self, nodes, index_map, writer, write_index, pb, output_lines=None):
        knit = KnitVersionedFiles(None, None)
        nodes = sorted(nodes)
        request_groups = {}
        for index, key, value in nodes:
            if index not in request_groups:
                request_groups[index] = []
            request_groups[index].append((key, value))
        record_index = 0
        pb.update('Copied record', record_index, len(nodes))
        for index, items in request_groups.items():
            pack_readv_requests = []
            for key, value in items:
                bits = value[1:].split(b' ')
                offset, length = (int(bits[0]), int(bits[1]))
                pack_readv_requests.append((offset, length, (key, value[0:1])))
            pack_readv_requests.sort()
            pack_obj = index_map[index]
            transport, path = pack_obj.access_tuple()
            try:
                reader = pack.make_readv_reader(transport, path, [offset[0:2] for offset in pack_readv_requests])
            except _mod_transport.NoSuchFile:
                if self._reload_func is not None:
                    self._reload_func()
                raise
            for (names, read_func), (_1, _2, (key, eol_flag)) in zip(reader.iter_records(), pack_readv_requests):
                raw_data = read_func(None)
                if output_lines is not None:
                    output_lines(knit._parse_record(key[-1], raw_data)[0])
                else:
                    df, _ = knit._parse_record_header(key, raw_data)
                    df.close()
                pos, size = writer.add_bytes_record([raw_data], len(raw_data), names)
                write_index.add_node(key, eol_flag + b'%d %d' % (pos, size))
                pb.update('Copied record', record_index)
                record_index += 1

    def _copy_nodes_graph(self, index_map, writer, write_index, readv_group_iter, total_items, output_lines=False):
        """Copy knit nodes between packs.

        :param output_lines: Return lines present in the copied data as
            an iterator of line,version_id.
        """
        with ui.ui_factory.nested_progress_bar() as pb:
            yield from self._do_copy_nodes_graph(index_map, writer, write_index, output_lines, pb, readv_group_iter, total_items)

    def _do_copy_nodes_graph(self, index_map, writer, write_index, output_lines, pb, readv_group_iter, total_items):
        knit = KnitVersionedFiles(None, None)
        if output_lines:
            factory = KnitPlainFactory()
        record_index = 0
        pb.update('Copied record', record_index, total_items)
        for index, readv_vector, node_vector in readv_group_iter:
            pack_obj = index_map[index]
            transport, path = pack_obj.access_tuple()
            try:
                reader = pack.make_readv_reader(transport, path, readv_vector)
            except _mod_transport.NoSuchFile:
                if self._reload_func is not None:
                    self._reload_func()
                raise
            for (names, read_func), (key, eol_flag, references) in zip(reader.iter_records(), node_vector):
                raw_data = read_func(None)
                if output_lines:
                    content, _ = knit._parse_record(key[-1], raw_data)
                    if len(references[-1]) == 0:
                        line_iterator = factory.get_fulltext_content(content)
                    else:
                        line_iterator = factory.get_linedelta_content(content)
                    for line in line_iterator:
                        yield (line, key)
                else:
                    df, _ = knit._parse_record_header(key, raw_data)
                    df.close()
                pos, size = writer.add_bytes_record([raw_data], len(raw_data), names)
                write_index.add_node(key, eol_flag + b'%d %d' % (pos, size), references)
                pb.update('Copied record', record_index)
                record_index += 1

    def _process_inventory_lines(self, inv_lines):
        """Use up the inv_lines generator and setup a text key filter."""
        repo = self._pack_collection.repo
        fileid_revisions = repo._find_file_ids_from_xml_inventory_lines(inv_lines, self.revision_keys)
        text_filter = []
        for fileid, file_revids in fileid_revisions.items():
            text_filter.extend([(fileid, file_revid) for file_revid in file_revids])
        self._text_filter = text_filter

    def _copy_inventory_texts(self):
        inv_keys = self._revision_keys
        inventory_index_map, inventory_indices = self._pack_map_and_index_list('inventory_index')
        inv_nodes = self._index_contents(inventory_indices, inv_keys)
        self.pb.update('Copying inventory texts', 2)
        total_items, readv_group_iter = self._least_readv_node_readv(inv_nodes)
        output_lines = bool(self.revision_ids)
        inv_lines = self._copy_nodes_graph(inventory_index_map, self.new_pack._writer, self.new_pack.inventory_index, readv_group_iter, total_items, output_lines=output_lines)
        if self.revision_ids:
            self._process_inventory_lines(inv_lines)
        else:
            list(inv_lines)
            self._text_filter = None
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: inventories copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.inventory_index.key_count(), time.time() - self.new_pack.start_time)

    def _update_pack_order(self, entries, index_to_pack_map):
        """Determine how we want our packs to be ordered.

        This changes the sort order of the self.packs list so that packs unused
        by 'entries' will be at the end of the list, so that future requests
        can avoid probing them.  Used packs will be at the front of the
        self.packs list, in the order of their first use in 'entries'.

        :param entries: A list of (index, ...) tuples
        :param index_to_pack_map: A mapping from index objects to pack objects.
        """
        packs = []
        seen_indexes = set()
        for entry in entries:
            index = entry[0]
            if index not in seen_indexes:
                packs.append(index_to_pack_map[index])
                seen_indexes.add(index)
        if len(packs) == len(self.packs):
            if 'pack' in debug.debug_flags:
                trace.mutter('Not changing pack list, all packs used.')
            return
        seen_packs = set(packs)
        for pack in self.packs:
            if pack not in seen_packs:
                packs.append(pack)
                seen_packs.add(pack)
        if 'pack' in debug.debug_flags:
            old_names = [p.access_tuple()[1] for p in self.packs]
            new_names = [p.access_tuple()[1] for p in packs]
            trace.mutter('Reordering packs\nfrom: %s\n  to: %s', old_names, new_names)
        self.packs = packs

    def _copy_revision_texts(self):
        if self.revision_ids:
            revision_keys = [(revision_id,) for revision_id in self.revision_ids]
        else:
            revision_keys = None
        revision_index_map, revision_indices = self._pack_map_and_index_list('revision_index')
        revision_nodes = self._index_contents(revision_indices, revision_keys)
        revision_nodes = list(revision_nodes)
        self._update_pack_order(revision_nodes, revision_index_map)
        self.pb.update('Copying revision texts', 1)
        total_items, readv_group_iter = self._revision_node_readv(revision_nodes)
        list(self._copy_nodes_graph(revision_index_map, self.new_pack._writer, self.new_pack.revision_index, readv_group_iter, total_items))
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: revisions copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.revision_index.key_count(), time.time() - self.new_pack.start_time)
        self._revision_keys = revision_keys

    def _get_text_nodes(self):
        text_index_map, text_indices = self._pack_map_and_index_list('text_index')
        return (text_index_map, self._index_contents(text_indices, self._text_filter))

    def _copy_text_texts(self):
        text_index_map, text_nodes = self._get_text_nodes()
        if self._text_filter is not None:
            text_nodes = set(text_nodes)
            present_text_keys = {_node[1] for _node in text_nodes}
            missing_text_keys = set(self._text_filter) - present_text_keys
            if missing_text_keys:
                trace.mutter('missing keys during fetch: %r', missing_text_keys)
                a_missing_key = missing_text_keys.pop()
                raise errors.RevisionNotPresent(a_missing_key[1], a_missing_key[0])
        self.pb.update('Copying content texts', 3)
        total_items, readv_group_iter = self._least_readv_node_readv(text_nodes)
        list(self._copy_nodes_graph(text_index_map, self.new_pack._writer, self.new_pack.text_index, readv_group_iter, total_items))
        self._log_copied_texts()

    def _create_pack_from_packs(self):
        self.pb.update('Opening pack', 0, 5)
        self.new_pack = self.open_pack()
        new_pack = self.new_pack
        new_pack.set_write_cache_size(1024 * 1024)
        if 'pack' in debug.debug_flags:
            plain_pack_list = ['{}{}'.format(a_pack.pack_transport.base, a_pack.name) for a_pack in self.packs]
            if self.revision_ids is not None:
                rev_count = len(self.revision_ids)
            else:
                rev_count = 'all'
            trace.mutter('%s: create_pack: creating pack from source packs: %s%s %s revisions wanted %s t=0', time.ctime(), self._pack_collection._upload_transport.base, new_pack.random_name, plain_pack_list, rev_count)
        self._copy_revision_texts()
        self._copy_inventory_texts()
        self._copy_text_texts()
        signature_filter = self._revision_keys
        signature_index_map, signature_indices = self._pack_map_and_index_list('signature_index')
        signature_nodes = self._index_contents(signature_indices, signature_filter)
        self.pb.update('Copying signature texts', 4)
        self._copy_nodes(signature_nodes, signature_index_map, new_pack._writer, new_pack.signature_index)
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: revision signatures copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, new_pack.random_name, new_pack.signature_index.key_count(), time.time() - new_pack.start_time)
        new_pack._check_references()
        if not self._use_pack(new_pack):
            new_pack.abort()
            return None
        self.pb.update('Finishing pack', 5)
        new_pack.finish()
        self._pack_collection.allocate(new_pack)
        return new_pack

    def _least_readv_node_readv(self, nodes):
        """Generate request groups for nodes using the least readv's.

        :param nodes: An iterable of graph index nodes.
        :return: Total node count and an iterator of the data needed to perform
            readvs to obtain the data for nodes. Each item yielded by the
            iterator is a tuple with:
            index, readv_vector, node_vector. readv_vector is a list ready to
            hand to the transport readv method, and node_vector is a list of
            (key, eol_flag, references) for the node retrieved by the
            matching readv_vector.
        """
        nodes = sorted(nodes)
        total = len(nodes)
        request_groups = {}
        for index, key, value, references in nodes:
            if index not in request_groups:
                request_groups[index] = []
            request_groups[index].append((key, value, references))
        result = []
        for index, items in request_groups.items():
            pack_readv_requests = []
            for key, value, references in items:
                bits = value[1:].split(b' ')
                offset, length = (int(bits[0]), int(bits[1]))
                pack_readv_requests.append(((offset, length), (key, value[0:1], references)))
            pack_readv_requests.sort()
            pack_readv = [readv for readv, node in pack_readv_requests]
            node_vector = [node for readv, node in pack_readv_requests]
            result.append((index, pack_readv, node_vector))
        return (total, result)

    def _revision_node_readv(self, revision_nodes):
        """Return the total revisions and the readv's to issue.

        :param revision_nodes: The revision index contents for the packs being
            incorporated into the new pack.
        :return: As per _least_readv_node_readv.
        """
        return self._least_readv_node_readv(revision_nodes)