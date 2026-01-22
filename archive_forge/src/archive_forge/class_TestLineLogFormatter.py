import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
class TestLineLogFormatter(TestCaseForLogFormatter):

    def test_line_log(self):
        """Line log should show revno

        bug #5162
        """
        wt = self.make_standard_commit('test-line-log', committer='Line-Log-Formatter Tester <test@line.log>', authors=[])
        self.assertFormatterResult(b'1: Line-Log-Formatte... 2005-11-22 add a\n', wt.branch, log.LineLogFormatter)

    def test_trailing_newlines(self):
        wt = self.make_branch_and_tree('.')
        b = self.make_commits_with_trailing_newlines(wt)
        self.assertFormatterResult(b'3: Joe Foo 2005-11-22 single line with trailing newline\n2: Joe Foo 2005-11-22 multiline\n1: Joe Foo 2005-11-22 simple log message\n', b, log.LineLogFormatter)

    def test_line_log_single_merge_revision(self):
        wt = self._prepare_tree_with_merges()
        revspec = revisionspec.RevisionSpec.from_string('1.1.1')
        rev = revspec.in_history(wt.branch)
        self.assertFormatterResult(b'1.1.1: Joe Foo 2005-11-22 rev-merged\n', wt.branch, log.LineLogFormatter, show_log_kwargs=dict(start_revision=rev, end_revision=rev))

    def test_line_log_with_tags(self):
        wt = self._prepare_tree_with_merges(with_tags=True)
        self.assertFormatterResult(b'3: Joe Foo 2005-11-22 {v1.0, v1.0rc1} rev-3\n2: Joe Foo 2005-11-22 [merge] {v0.2} rev-2\n1: Joe Foo 2005-11-22 rev-1\n', wt.branch, log.LineLogFormatter)