import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_format(self):
    tree = self.setup_ab_tree()
    start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
    end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
    self.assertFormatterResult(b'      Joe Foo\t2005-11-22\n      revision-id:3a\n      commit 3a\n\n      Joe Foo\t2005-11-22\n      revision-id:2a\n      commit 2a\n\n    1 Joe Foo\t2005-11-22\n      commit 1a\n\n', tree.branch, log.ShortLogFormatter, show_log_kwargs={'start_revision': start_rev, 'end_revision': end_rev})