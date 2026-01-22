from urllib.parse import urlsplit
from ... import debug, errors, trace, transport
from ...i18n import gettext
from ...urlutils import InvalidURL, split, join
from .account import get_lp_login
from .uris import DEFAULT_INSTANCE, LAUNCHPAD_DOMAINS, LPNET_SERVICE_ROOT
def _resolve_via_api(path, url, api_base_url=LPNET_SERVICE_ROOT):
    from .lp_api import connect_launchpad
    lp = connect_launchpad(api_base_url, version='devel')
    subpaths = []
    lp_branch = None
    git_repo = None
    while path:
        lp_branch = lp.branches.getByPath(path=path)
        git_repo = lp.git_repositories.getByPath(path=path)
        if lp_branch and git_repo:
            target = git_repo.target
            vcs = target.vcs
            trace.warning("Found both a Bazaar branch and a git repository at lp:%s. Using %s, since that is the projects' default vcs", path, vcs)
            if vcs == 'Git':
                lp_branch = None
            elif vcs == 'Bazaar':
                git_repo = None
            else:
                raise errors.BzrError('Unknown default vcs %s for %s' % (vcs, target))
        if lp_branch or git_repo:
            break
        path, subpath = split(path)
        subpaths.insert(0, subpath)
    if lp_branch:
        return {'urls': [join(lp_branch.composePublicURL(scheme='bzr+ssh'), *subpaths), join(lp_branch.composePublicURL(scheme='http'), *subpaths)]}
    elif git_repo:
        return {'urls': [join(git_repo.git_ssh_url, *subpaths), join(git_repo.git_https_url, *subpaths)]}
    else:
        raise InvalidURL(f'Unknown Launchpad path: {url}')