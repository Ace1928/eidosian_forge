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
def _copy_text_texts(self):
    """generate what texts we should have and then copy."""
    self.pb.update('Copying content texts', 3)
    repo = self._pack_collection.repo
    ancestors = {key[0]: tuple((ref[0] for ref in refs[0])) for _1, key, _2, refs in self.new_pack.revision_index.iter_all_entries()}
    ideal_index = repo._generate_text_key_index(self._text_refs, ancestors)
    ok_nodes = []
    bad_texts = []
    discarded_nodes = []
    NULL_REVISION = _mod_revision.NULL_REVISION
    text_index_map, text_nodes = self._get_text_nodes()
    for node in text_nodes:
        try:
            ideal_parents = tuple(ideal_index[node[1]])
        except KeyError:
            discarded_nodes.append(node)
            self._data_changed = True
        else:
            if ideal_parents == (NULL_REVISION,):
                ideal_parents = ()
            if ideal_parents == node[3][0]:
                ok_nodes.append(node)
            elif ideal_parents[0:1] == node[3][0][0:1]:
                self._data_changed = True
                ok_nodes.append((node[0], node[1], node[2], (ideal_parents, node[3][1])))
                self._data_changed = True
            else:
                bad_texts.append((node[1], ideal_parents))
                self._data_changed = True
    del ideal_index
    del text_nodes
    total_items, readv_group_iter = self._least_readv_node_readv(ok_nodes)
    list(self._copy_nodes_graph(text_index_map, self.new_pack._writer, self.new_pack.text_index, readv_group_iter, total_items))
    topo_order = tsort.topo_sort(ancestors)
    rev_order = dict(zip(topo_order, range(len(topo_order))))
    bad_texts.sort(key=lambda key: rev_order.get(key[0][1], 0))
    transaction = repo.get_transaction()
    file_id_index = GraphIndexPrefixAdapter(self.new_pack.text_index, ('blank',), 1, add_nodes_callback=self.new_pack.text_index.add_nodes)
    data_access = _DirectPackAccess({self.new_pack.text_index: self.new_pack.access_tuple()})
    data_access.set_writer(self.new_pack._writer, self.new_pack.text_index, self.new_pack.access_tuple())
    output_texts = KnitVersionedFiles(_KnitGraphIndex(self.new_pack.text_index, add_callback=self.new_pack.text_index.add_nodes, deltas=True, parents=True, is_locked=repo.is_locked), data_access=data_access, max_delta_chain=200)
    for key, parent_keys in bad_texts:
        self.new_pack.flush()
        parents = []
        for parent_key in parent_keys:
            if parent_key[0] != key[0]:
                raise errors.BzrError('Mismatched key parent %r:%r' % (key, parent_keys))
            parents.append(parent_key[1])
        text_lines = next(repo.texts.get_record_stream([key], 'unordered', True)).get_bytes_as('lines')
        output_texts.add_lines(key, parent_keys, text_lines, random_id=True, check_content=False)
    missing_text_keys = self.new_pack.text_index._external_references()
    if missing_text_keys:
        raise errors.BzrCheckError('Reference to missing compression parents %r' % (missing_text_keys,))
    self._log_copied_texts()