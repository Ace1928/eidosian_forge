from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_pack_collection_sets_sibling_indices(self):
    """The CombinedGraphIndex objects in the pack collection are all
        siblings of each other, so that search-order reorderings will be copied
        to each other.
        """
    repo = self.make_repository('repo')
    pack_coll = repo._pack_collection
    indices = {pack_coll.revision_index, pack_coll.inventory_index, pack_coll.text_index, pack_coll.signature_index}
    if pack_coll.chk_index is not None:
        indices.add(pack_coll.chk_index)
    combined_indices = {idx.combined_index for idx in indices}
    for combined_index in combined_indices:
        self.assertEqual(combined_indices.difference([combined_index]), combined_index._sibling_indices)