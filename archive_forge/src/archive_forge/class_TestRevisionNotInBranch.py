import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestRevisionNotInBranch(TestCaseForLogFormatter):

    def setup_a_tree(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        kwargs = {'committer': 'Joe Foo <joe@foo.com>', 'timestamp': 1132617600, 'timezone': 0}
        tree.commit('commit 1a', rev_id=b'1a', **kwargs)
        tree.commit('commit 2a', rev_id=b'2a', **kwargs)
        tree.commit('commit 3a', rev_id=b'3a', **kwargs)
        return tree

    def setup_ab_tree(self):
        tree = self.setup_a_tree()
        tree.set_last_revision(b'1a')
        tree.branch.set_last_revision_info(1, b'1a')
        kwargs = {'committer': 'Joe Foo <joe@foo.com>', 'timestamp': 1132617600, 'timezone': 0}
        tree.commit('commit 2b', rev_id=b'2b', **kwargs)
        tree.commit('commit 3b', rev_id=b'3b', **kwargs)
        return tree

    def test_one_revision(self):
        tree = self.setup_ab_tree()
        lf = LogCatcher()
        rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
        log.show_log(tree.branch, lf, verbose=True, start_revision=rev, end_revision=rev)
        self.assertEqual(1, len(lf.revisions))
        self.assertEqual(None, lf.revisions[0].revno)
        self.assertEqual(b'3a', lf.revisions[0].rev.revision_id)

    def test_many_revisions(self):
        tree = self.setup_ab_tree()
        lf = LogCatcher()
        start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
        end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
        log.show_log(tree.branch, lf, verbose=True, start_revision=start_rev, end_revision=end_rev)
        self.assertEqual(3, len(lf.revisions))
        self.assertEqual(None, lf.revisions[0].revno)
        self.assertEqual(b'3a', lf.revisions[0].rev.revision_id)
        self.assertEqual(None, lf.revisions[1].revno)
        self.assertEqual(b'2a', lf.revisions[1].rev.revision_id)
        self.assertEqual('1', lf.revisions[2].revno)

    def test_long_format(self):
        tree = self.setup_ab_tree()
        start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
        end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
        self.assertFormatterResult(b'------------------------------------------------------------\nrevision-id: 3a\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: tree\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  commit 3a\n------------------------------------------------------------\nrevision-id: 2a\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: tree\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  commit 2a\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: tree\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  commit 1a\n', tree.branch, log.LongLogFormatter, show_log_kwargs={'start_revision': start_rev, 'end_revision': end_rev})

    def test_short_format(self):
        tree = self.setup_ab_tree()
        start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
        end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
        self.assertFormatterResult(b'      Joe Foo\t2005-11-22\n      revision-id:3a\n      commit 3a\n\n      Joe Foo\t2005-11-22\n      revision-id:2a\n      commit 2a\n\n    1 Joe Foo\t2005-11-22\n      commit 1a\n\n', tree.branch, log.ShortLogFormatter, show_log_kwargs={'start_revision': start_rev, 'end_revision': end_rev})

    def test_line_format(self):
        tree = self.setup_ab_tree()
        start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
        end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
        self.assertFormatterResult(b'Joe Foo 2005-11-22 commit 3a\nJoe Foo 2005-11-22 commit 2a\n1: Joe Foo 2005-11-22 commit 1a\n', tree.branch, log.LineLogFormatter, show_log_kwargs={'start_revision': start_rev, 'end_revision': end_rev})