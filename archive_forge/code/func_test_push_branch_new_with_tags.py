import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
def test_push_branch_new_with_tags(self):
    remote = ControlDir.open(self.remote_url)
    builder = self.make_branch_builder('local', format=self._from_format)
    builder.start_series()
    rev_1 = builder.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('filename', None, 'file', b'content'))])
    rev_2 = builder.build_snapshot([rev_1], [('modify', ('filename', b'new-content\n'))])
    rev_3 = builder.build_snapshot([rev_1], [('modify', ('filename', b'new-new-content\n'))])
    builder.finish_series()
    branch = builder.get_branch()
    try:
        branch.tags.set_tag('atag', rev_2)
    except TagsNotSupported:
        raise TestNotApplicable('source format does not support tags')
    branch.get_config_stack().set('branch.fetch_tags', True)
    if self._from_format == 'git':
        result = remote.push_branch(branch, name='newbranch')
    else:
        result = remote.push_branch(branch, lossy=True, name='newbranch')
    self.assertEqual(0, result.old_revno)
    if self._from_format == 'git':
        self.assertEqual(2, result.new_revno)
    else:
        self.assertIs(None, result.new_revno)
    result.report(BytesIO())
    self.assertEqual({b'refs/heads/newbranch', b'refs/tags/atag'}, set(self.remote_real.get_refs().keys()))