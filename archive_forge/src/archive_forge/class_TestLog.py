import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLog(tests.TestCaseWithTransport, test_log.TestLogMixin):

    def make_minimal_branch(self, path='.', format=None):
        tree = self.make_branch_and_tree(path, format=format)
        self.build_tree([path + '/hello.txt'])
        tree.add('hello.txt')
        tree.commit(message='message1')
        return tree

    def make_linear_branch(self, path='.', format=None):
        tree = self.make_branch_and_tree(path, format=format)
        self.build_tree([path + '/hello.txt', path + '/goodbye.txt', path + '/meep.txt'])
        tree.add('hello.txt')
        tree.commit(message='message1')
        tree.add('goodbye.txt')
        tree.commit(message='message2')
        tree.add('meep.txt')
        tree.commit(message='message3')
        return tree

    def make_merged_branch(self, path='.', format=None):
        tree = self.make_linear_branch(path, format)
        tree2 = tree.controldir.sprout('tree2', revision_id=tree.branch.get_rev_id(1)).open_workingtree()
        tree2.commit(message='tree2 message2')
        tree2.commit(message='tree2 message3')
        tree.merge_from_branch(tree2.branch)
        tree.commit(message='merge')
        return tree