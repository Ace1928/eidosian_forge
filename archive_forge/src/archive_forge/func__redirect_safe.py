import os
import re
import uuid
from urllib.parse import urlparse
from tornado.escape import url_escape
from ..base.handlers import JupyterHandler
from .decorator import allow_unauthenticated
from .security import passwd_check, set_password
def _redirect_safe(self, url, default=None):
    """Redirect if url is on our PATH

        Full-domain redirects are allowed if they pass our CORS origin checks.

        Otherwise use default (self.base_url if unspecified).
        """
    if default is None:
        default = self.base_url
    url = url.replace('\\', '%5C')
    if ':' in url:
        scheme, _, rest = url.partition(':')
        url = f'{scheme}://{rest.lstrip('/')}'
    parsed = urlparse(url)
    if (parsed.scheme or parsed.netloc) or not (parsed.path + '/').startswith(self.base_url):
        allow = False
        if parsed.scheme or parsed.netloc:
            origin = f'{parsed.scheme}://{parsed.netloc}'
            origin = origin.lower()
            if self.allow_origin:
                allow = self.allow_origin == origin
            elif self.allow_origin_pat:
                allow = bool(re.match(self.allow_origin_pat, origin))
        if not allow:
            self.log.warning('Not allowing login redirect to %r' % url)
            url = default
    self.redirect(url)