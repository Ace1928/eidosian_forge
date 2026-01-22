import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def _prepare_tree_with_merges(self, with_tags=False):
    wt = self.make_branch_and_memory_tree('.')
    wt.lock_write()
    self.addCleanup(wt.unlock)
    wt.add('')
    self.wt_commit(wt, 'rev-1', rev_id=b'rev-1')
    self.wt_commit(wt, 'rev-merged', rev_id=b'rev-2a')
    wt.set_parent_ids([b'rev-1', b'rev-2a'])
    wt.branch.set_last_revision_info(1, b'rev-1')
    self.wt_commit(wt, 'rev-2', rev_id=b'rev-2b')
    if with_tags:
        branch = wt.branch
        branch.tags.set_tag('v0.2', b'rev-2b')
        self.wt_commit(wt, 'rev-3', rev_id=b'rev-3')
        branch.tags.set_tag('v1.0rc1', b'rev-3')
        branch.tags.set_tag('v1.0', b'rev-3')
    return wt