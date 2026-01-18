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
def test_pack_layout(self):
    format = self.get_format()
    tree = self.make_branch_and_tree('.', format=format)
    trans = tree.branch.repository.controldir.get_repository_transport(None)
    tree.commit('start', rev_id=b'1')
    tree.commit('more work', rev_id=b'2')
    tree.branch.repository.pack()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    pack = tree.branch.repository._pack_collection.get_pack_by_name(tree.branch.repository._pack_collection.names()[0])
    pos_1 = pos_2 = None
    for _1, key, val, refs in pack.revision_index.iter_all_entries():
        if isinstance(format.repository_format, RepositoryFormat2a):
            pos = list(map(int, val.split()))
        else:
            pos = int(val[1:].split()[0])
        if key == (b'1',):
            pos_1 = pos
        else:
            pos_2 = pos
    self.assertTrue(pos_2 < pos_1, 'rev 1 came before rev 2 %s > %s' % (pos_1, pos_2))