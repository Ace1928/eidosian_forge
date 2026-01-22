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
def _show_repository_info(repository, outfile):
    """Show settings of a repository."""
    if repository.make_working_trees():
        outfile.write('\n')
        outfile.write('Create working tree for new branches inside the repository.\n')