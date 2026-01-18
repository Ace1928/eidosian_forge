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
def test_pulling_nothing_leads_to_no_new_names(self):
    format = self.get_format()
    tree1 = self.make_branch_and_tree('1', format=format)
    tree2 = self.make_branch_and_tree('2', format=format)
    tree1.branch.repository.fetch(tree2.branch.repository)
    trans = tree1.branch.repository.controldir.get_repository_transport(None)
    self.assertEqual([], list(self.index_class(trans, 'pack-names', None).iter_all_entries()))