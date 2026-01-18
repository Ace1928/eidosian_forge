import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
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