import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_nonstrict_commit_without_unknowns(self):
    """Try and commit with no unknown files and strict = False,
        should work."""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    with open('hello', 'w') as f:
        f.write('hello world')
    wt.add('hello')
    wt.commit(message='add hello', strict=False)