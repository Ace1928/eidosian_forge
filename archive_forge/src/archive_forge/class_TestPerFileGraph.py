from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
class TestPerFileGraph(TestCaseWithRepository):

    def test_file_graph(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('a', b'contents')])
        tree.add(['a'])
        if not tree.supports_file_ids:
            raise TestNotApplicable('file ids not supported')
        fileid = tree.path2id('a')
        revid1 = tree.commit('msg')
        self.build_tree_contents([('a', b'new contents')])
        revid2 = tree.commit('msg')
        self.addCleanup(tree.lock_read().unlock)
        graph = tree.branch.repository.get_file_graph()
        self.assertEqual({(fileid, revid2): ((fileid, revid1),), (fileid, revid1): ()}, graph.get_parent_map([(fileid, revid2), (fileid, revid1)]))