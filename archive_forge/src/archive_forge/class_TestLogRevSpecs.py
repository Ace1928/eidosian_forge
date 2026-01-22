import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogRevSpecs(TestLogWithLogCatcher):

    def test_log_no_revspec(self):
        self.make_linear_branch()
        self.assertLogRevnos([], ['3', '2', '1'])

    def test_log_null_end_revspec(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r1..'], ['3', '2', '1'])

    def test_log_null_begin_revspec(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r..3'], ['3', '2', '1'])

    def test_log_null_both_revspecs(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r..'], ['3', '2', '1'])

    def test_log_negative_begin_revspec_full_log(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r-3..'], ['3', '2', '1'])

    def test_log_negative_both_revspec_full_log(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r-3..-1'], ['3', '2', '1'])

    def test_log_negative_both_revspec_partial(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r-3..-2'], ['2', '1'])

    def test_log_negative_begin_revspec(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r-2..'], ['3', '2'])

    def test_log_positive_revspecs(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-r1..3'], ['3', '2', '1'])

    def test_log_dotted_revspecs(self):
        self.make_merged_branch()
        self.assertLogRevnos(['-n0', '-r1..1.1.1'], ['1.1.1', '1'])

    def test_log_limit(self):
        tree = self.make_branch_and_tree('.')
        for pos in range(10):
            tree.commit('%s' % pos)
        self.assertLogRevnos(['--limit', '2'], ['10', '9'])

    def test_log_limit_short(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-l', '2'], ['3', '2'])

    def test_log_change_revno(self):
        self.make_linear_branch()
        self.assertLogRevnos(['-c1'], ['1'])

    def test_branch_revspec(self):
        foo = self.make_branch_and_tree('foo')
        bar = self.make_branch_and_tree('bar')
        self.build_tree(['foo/foo.txt', 'bar/bar.txt'])
        foo.add('foo.txt')
        bar.add('bar.txt')
        foo.commit(message='foo')
        bar.commit(message='bar')
        self.run_bzr('log -r branch:../bar', working_dir='foo')
        self.assertEqual([bar.branch.get_rev_id(1)], [r.rev.revision_id for r in self.get_captured_revisions()])