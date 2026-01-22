import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _handle_http_ecp_redirect(self, session, response, method, **kwargs):
    if response.status_code not in (self.HTTP_MOVED_TEMPORARILY, self.HTTP_SEE_OTHER):
        return response
    location = response.headers['location']
    return session.request(location, method, authenticated=False, **kwargs)