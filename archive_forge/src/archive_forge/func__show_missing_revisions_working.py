import sys
import time
from io import StringIO
from . import branch as _mod_branch
from . import controldir, errors
from . import hooks as _mod_hooks
from . import osutils, urlutils
from .bzr import bzrdir
from .errors import (NoRepositoryPresent, NotBranchError, NotLocalUrl,
from .missing import find_unmerged
def _show_missing_revisions_working(working, outfile):
    """Show missing revisions in working tree."""
    branch = working.branch
    try:
        branch_revno, branch_last_revision = branch.last_revision_info()
    except errors.UnsupportedOperation:
        return
    try:
        tree_last_id = working.get_parent_ids()[0]
    except IndexError:
        tree_last_id = None
    if branch_revno and tree_last_id != branch_last_revision:
        tree_last_revno = branch.revision_id_to_revno(tree_last_id)
        missing_count = branch_revno - tree_last_revno
        outfile.write('\n')
        outfile.write('Working tree is out of date: missing %d revision%s.\n' % (missing_count, plural(missing_count)))