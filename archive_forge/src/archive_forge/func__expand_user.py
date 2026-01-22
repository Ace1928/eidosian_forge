from urllib.parse import urlsplit
from ... import debug, errors, trace, transport
from ...i18n import gettext
from ...urlutils import InvalidURL, split, join
from .account import get_lp_login
from .uris import DEFAULT_INSTANCE, LAUNCHPAD_DOMAINS, LPNET_SERVICE_ROOT
def _expand_user(path, url, lp_login):
    if path.startswith('~/'):
        if lp_login is None:
            raise InvalidURL(path=url, extra='Cannot resolve "~" to your username. See "bzr help launchpad-login"')
        path = '~' + lp_login + path[1:]
    return path