import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_short_log_with_merges_and_range(self):
    wt = self._prepare_tree_with_merges()
    self.wt_commit(wt, 'rev-3a', rev_id=b'rev-3a')
    wt.branch.set_last_revision_info(2, b'rev-2b')
    wt.set_parent_ids([b'rev-2b', b'rev-3a'])
    self.wt_commit(wt, 'rev-3b', rev_id=b'rev-3b')
    self.assertFormatterResult(b'    3 Joe Foo\t2005-11-22 [merge]\n      rev-3b\n\n    2 Joe Foo\t2005-11-22 [merge]\n      rev-2\n\n', wt.branch, log.ShortLogFormatter, show_log_kwargs=dict(start_revision=2, end_revision=3))