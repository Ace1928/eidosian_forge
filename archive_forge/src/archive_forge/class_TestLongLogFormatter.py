import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLongLogFormatter(TestCaseForLogFormatter):

    def test_verbose_log(self):
        """Verbose log includes changed files

        bug #4676
        """
        wt = self.make_standard_commit('test_verbose_log', authors=[])
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_verbose_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\nadded:\n  a\n', wt.branch, log.LongLogFormatter, show_log_kwargs=dict(verbose=True))

    def test_merges_are_indented_by_level(self):
        wt = self.make_branch_and_tree('parent')
        self.wt_commit(wt, 'first post')
        child_wt = wt.controldir.sprout('child').open_workingtree()
        self.wt_commit(child_wt, 'branch 1')
        smallerchild_wt = wt.controldir.sprout('smallerchild').open_workingtree()
        self.wt_commit(smallerchild_wt, 'branch 2')
        child_wt.merge_from_branch(smallerchild_wt.branch)
        self.wt_commit(child_wt, 'merge branch 2')
        wt.merge_from_branch(child_wt.branch)
        self.wt_commit(wt, 'merge branch 1')
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 2 [merge]\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:04 +0000\nmessage:\n  merge branch 1\n    ------------------------------------------------------------\n    revno: 1.1.2 [merge]\n    committer: Joe Foo <joe@foo.com>\n    branch nick: child\n    timestamp: Tue 2005-11-22 00:00:03 +0000\n    message:\n      merge branch 2\n        ------------------------------------------------------------\n        revno: 1.2.1\n        committer: Joe Foo <joe@foo.com>\n        branch nick: smallerchild\n        timestamp: Tue 2005-11-22 00:00:02 +0000\n        message:\n          branch 2\n    ------------------------------------------------------------\n    revno: 1.1.1\n    committer: Joe Foo <joe@foo.com>\n    branch nick: child\n    timestamp: Tue 2005-11-22 00:00:01 +0000\n    message:\n      branch 1\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  first post\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=0), show_log_kwargs=dict(verbose=True))

    def test_verbose_merge_revisions_contain_deltas(self):
        wt = self.make_branch_and_tree('parent')
        self.build_tree(['parent/f1', 'parent/f2'])
        wt.add(['f1', 'f2'])
        self.wt_commit(wt, 'first post')
        child_wt = wt.controldir.sprout('child').open_workingtree()
        os.unlink('child/f1')
        self.build_tree_contents([('child/f2', b'hello\n')])
        self.wt_commit(child_wt, 'removed f1 and modified f2')
        wt.merge_from_branch(child_wt.branch)
        self.wt_commit(wt, 'merge branch 1')
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 2 [merge]\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:02 +0000\nmessage:\n  merge branch 1\nremoved:\n  f1\nmodified:\n  f2\n    ------------------------------------------------------------\n    revno: 1.1.1\n    committer: Joe Foo <joe@foo.com>\n    branch nick: child\n    timestamp: Tue 2005-11-22 00:00:01 +0000\n    message:\n      removed f1 and modified f2\n    removed:\n      f1\n    modified:\n      f2\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  first post\nadded:\n  f1\n  f2\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=0), show_log_kwargs=dict(verbose=True))

    def test_trailing_newlines(self):
        wt = self.make_branch_and_tree('.')
        b = self.make_commits_with_trailing_newlines(wt)
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 3\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:02 +0000\nmessage:\n  single line with trailing newline\n------------------------------------------------------------\nrevno: 2\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:01 +0000\nmessage:\n  multiline\n  log\n  message\n------------------------------------------------------------\nrevno: 1\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: test\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  simple log message\n', b, log.LongLogFormatter)

    def test_author_in_log(self):
        """Log includes the author name if it's set in
        the revision properties
        """
        wt = self.make_standard_commit('test_author_log', authors=['John Doe <jdoe@example.com>', 'Jane Rey <jrey@example.com>'])
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\nauthor: John Doe <jdoe@example.com>, Jane Rey <jrey@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_author_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n', wt.branch, log.LongLogFormatter)

    def test_properties_in_log(self):
        """Log includes the custom properties returned by the registered
        handlers.
        """
        wt = self.make_standard_commit('test_properties_in_log')

        def trivial_custom_prop_handler(revision):
            return {'test_prop': 'test_value'}
        log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 1\ntest_prop: test_value\nauthor: John Doe <jdoe@example.com>\ncommitter: Lorem Ipsum <test@example.com>\nbranch nick: test_properties_in_log\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  add a\n', wt.branch, log.LongLogFormatter)

    def test_properties_in_short_log(self):
        """Log includes the custom properties returned by the registered
        handlers.
        """
        wt = self.make_standard_commit('test_properties_in_short_log')

        def trivial_custom_prop_handler(revision):
            return {'test_prop': 'test_value'}
        log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
        self.assertFormatterResult(b'    1 John Doe\t2005-11-22\n      test_prop: test_value\n      add a\n\n', wt.branch, log.ShortLogFormatter)

    def test_error_in_properties_handler(self):
        """Log includes the custom properties returned by the registered
        handlers.
        """
        wt = self.make_standard_commit('error_in_properties_handler', revprops={'first_prop': 'first_value'})
        sio = self.make_utf8_encoded_stringio()
        formatter = log.LongLogFormatter(to_file=sio)

        def trivial_custom_prop_handler(revision):
            raise Exception('a test error')
        log.properties_handler_registry.register('trivial_custom_prop_handler', trivial_custom_prop_handler)
        log.show_log(wt.branch, formatter)
        self.assertContainsRe(sio.getvalue(), b'brz: ERROR: Exception: a test error')

    def test_properties_handler_bad_argument(self):
        wt = self.make_standard_commit('bad_argument', revprops={'a_prop': 'test_value'})
        sio = self.make_utf8_encoded_stringio()
        formatter = log.LongLogFormatter(to_file=sio)

        def bad_argument_prop_handler(revision):
            return {'custom_prop_name': revision.properties['a_prop']}
        log.properties_handler_registry.register('bad_argument_prop_handler', bad_argument_prop_handler)
        self.assertRaises(AttributeError, formatter.show_properties, 'a revision', '')
        revision = wt.branch.repository.get_revision(wt.branch.last_revision())
        formatter.show_properties(revision, '')
        self.assertEqualDiff(b'custom_prop_name: test_value\n', sio.getvalue())

    def test_show_ids(self):
        wt = self.make_branch_and_tree('parent')
        self.build_tree(['parent/f1', 'parent/f2'])
        wt.add(['f1', 'f2'])
        self.wt_commit(wt, 'first post', rev_id=b'a')
        child_wt = wt.controldir.sprout('child').open_workingtree()
        self.wt_commit(child_wt, 'branch 1 changes', rev_id=b'b')
        wt.merge_from_branch(child_wt.branch)
        self.wt_commit(wt, 'merge branch 1', rev_id=b'c')
        self.assertFormatterResult(b'------------------------------------------------------------\nrevno: 2 [merge]\nrevision-id: c\nparent: a\nparent: b\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:02 +0000\nmessage:\n  merge branch 1\n    ------------------------------------------------------------\n    revno: 1.1.1\n    revision-id: b\n    parent: a\n    committer: Joe Foo <joe@foo.com>\n    branch nick: child\n    timestamp: Tue 2005-11-22 00:00:01 +0000\n    message:\n      branch 1 changes\n------------------------------------------------------------\nrevno: 1\nrevision-id: a\ncommitter: Joe Foo <joe@foo.com>\nbranch nick: parent\ntimestamp: Tue 2005-11-22 00:00:00 +0000\nmessage:\n  first post\n', wt.branch, log.LongLogFormatter, formatter_kwargs=dict(levels=0, show_ids=True))