from urllib.parse import urlsplit
from ... import debug, errors, trace, transport
from ...i18n import gettext
from ...urlutils import InvalidURL, split, join
from .account import get_lp_login
from .uris import DEFAULT_INSTANCE, LAUNCHPAD_DOMAINS, LPNET_SERVICE_ROOT
class LaunchpadDirectory:

    def look_up(self, name, url, purpose=None):
        """See DirectoryService.look_up"""
        return _resolve(url)