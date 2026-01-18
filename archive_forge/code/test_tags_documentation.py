from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
'brz merge' alone does not propagate tags to a master branch.

        (If the user runs 'brz commit', then that is when the tags from the
        merge are propagated.)
        