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
def _show_repository_stats(repository, stats, outfile):
    """Show statistics about a repository."""
    f = StringIO()
    if 'revisions' in stats:
        revisions = stats['revisions']
        f.write('  %8d revision%s\n' % (revisions, plural(revisions)))
    if 'size' in stats:
        f.write('  %8d KiB\n' % (stats['size'] / 1024))
    for hook in hooks['repository']:
        hook(repository, stats, f)
    if f.getvalue() != '':
        outfile.write('\n')
        outfile.write('Repository:\n')
        outfile.write(f.getvalue())