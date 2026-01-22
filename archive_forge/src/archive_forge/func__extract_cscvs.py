from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def _extract_cscvs(rev):
    """Older-style launchpad-cscvs import."""
    if 'cscvs-svn-branch-path' not in rev.properties:
        return
    yield ('svn', '{}:{}:{}'.format(rev.properties['cscvs-svn-repository-uuid'], rev.properties['cscvs-svn-revision-number'], urlutils.quote(rev.properties['cscvs-svn-branch-path'].strip('/'))))