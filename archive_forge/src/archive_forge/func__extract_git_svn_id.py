from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
def _extract_git_svn_id(rev):
    if 'git-svn-id' not in rev.properties:
        return
    full_url, revnum, uuid = parse_git_svn_id(rev.properties['git-svn-id'])
    branch_path = svn_branch_path_finder.find_branch_path(uuid, full_url)
    if branch_path is not None:
        yield ('svn', '%s:%d:%s' % (uuid, revnum, urlutils.quote(branch_path)))