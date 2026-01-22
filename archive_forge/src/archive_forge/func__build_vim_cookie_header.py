import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
def _build_vim_cookie_header(self, verify_session=False):
    """Build ESX host session cookie header."""
    if verify_session and (not self.session.is_current_session_active()):
        self.reset_session()
    vim_cookies = self.session.vim.client.cookiejar
    if len(list(vim_cookies)) > 0:
        cookie = list(vim_cookies)[0]
        return cookie.name + '=' + cookie.value