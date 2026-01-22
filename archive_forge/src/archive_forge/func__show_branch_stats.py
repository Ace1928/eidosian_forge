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
def _show_branch_stats(branch, verbose, outfile):
    """Show statistics about a branch."""
    try:
        revno, head = branch.last_revision_info()
    except errors.UnsupportedOperation:
        return {}
    outfile.write('\n')
    outfile.write('Branch history:\n')
    outfile.write('  %8d revision%s\n' % (revno, plural(revno)))
    stats = branch.repository.gather_stats(head, committers=verbose)
    if verbose:
        committers = stats['committers']
        outfile.write('  %8d committer%s\n' % (committers, plural(committers)))
    if revno:
        timestamp, timezone = stats['firstrev']
        age = int((time.time() - timestamp) / 3600 / 24)
        outfile.write('  %8d day%s old\n' % (age, plural(age)))
        outfile.write('   first revision: %s\n' % osutils.format_date(timestamp, timezone))
        timestamp, timezone = stats['latestrev']
        outfile.write('  latest revision: %s\n' % osutils.format_date(timestamp, timezone))
    return stats