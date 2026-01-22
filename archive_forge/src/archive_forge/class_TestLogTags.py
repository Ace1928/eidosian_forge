import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogTags(TestLog):

    def test_log_with_tags(self):
        tree = self.make_linear_branch(format='dirstate-tags')
        branch = tree.branch
        branch.tags.set_tag('tag1', branch.get_rev_id(1))
        branch.tags.set_tag('tag1.1', branch.get_rev_id(1))
        branch.tags.set_tag('tag3', branch.last_revision())
        log = self.run_bzr('log -r-1')[0]
        self.assertTrue('tags: tag3' in log)
        log = self.run_bzr('log -r1')[0]
        self.assertContainsRe(log, 'tags: (tag1, tag1\\.1|tag1\\.1, tag1)')

    def test_merged_log_with_tags(self):
        branch1_tree = self.make_linear_branch('branch1', format='dirstate-tags')
        branch1 = branch1_tree.branch
        branch2_tree = branch1_tree.controldir.sprout('branch2').open_workingtree()
        branch1_tree.commit(message='foobar', allow_pointless=True)
        branch1.tags.set_tag('tag1', branch1.last_revision())
        self.run_bzr('merge ../branch1', working_dir='branch2')
        branch2_tree.commit(message='merge branch 1')
        log = self.run_bzr('log -n0 -r-1', working_dir='branch2')[0]
        self.assertContainsRe(log, '    tags: tag1')
        log = self.run_bzr('log -n0 -r3.1.1', working_dir='branch2')[0]
        self.assertContainsRe(log, 'tags: tag1')