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
def test_partial_commit_move(self):
    """Test a partial commit where a file was renamed but not committed.

        https://bugs.launchpad.net/bzr/+bug/83039

        If not handled properly, commit will try to snapshot
        dialog.py with olive/ as a parent, while
        olive/ has not been snapshotted yet.
        """
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    self.build_tree(['annotate/', 'annotate/foo.py', 'olive/', 'olive/dialog.py'])
    wt.add(['annotate', 'olive', 'annotate/foo.py', 'olive/dialog.py'])
    wt.commit(message='add files')
    wt.rename_one('olive/dialog.py', 'aaa')
    self.build_tree_contents([('annotate/foo.py', b'modified\n')])
    wt.commit('renamed hello', specific_files=['annotate'])