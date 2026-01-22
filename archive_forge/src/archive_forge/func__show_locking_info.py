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
def _show_locking_info(repository=None, branch=None, working=None, outfile=None):
    """Show locking status of working, branch and repository."""
    if repository and repository.get_physical_lock_status() or (branch and branch.get_physical_lock_status()) or (working and working.get_physical_lock_status()):
        outfile.write('\n')
        outfile.write('Lock status:\n')
        if working:
            if working.get_physical_lock_status():
                status = 'locked'
            else:
                status = 'unlocked'
            outfile.write('  working tree: %s\n' % status)
        if branch:
            if branch.get_physical_lock_status():
                status = 'locked'
            else:
                status = 'unlocked'
            outfile.write('        branch: %s\n' % status)
        if repository:
            if repository.get_physical_lock_status():
                status = 'locked'
            else:
                status = 'unlocked'
            outfile.write('    repository: %s\n' % status)