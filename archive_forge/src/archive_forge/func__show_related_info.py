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
def _show_related_info(branch, outfile):
    """Show parent and push location of branch."""
    locs = _gather_related_branches(branch)
    if len(locs.locs) > 0:
        outfile.write('\n')
        outfile.write('Related branches:\n')
        outfile.writelines(locs.get_lines())