import datetime
import urllib.parse
import uuid
from lxml import etree  # nosec(cjschaef): used to create xml, not parse it
from oslo_config import cfg
from keystoneclient import access
from keystoneclient.auth.identity import v3
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _prepare_idp_saml2_request(self, saml2_authn_request):
    header = saml2_authn_request[self.SAML2_HEADER_INDEX]
    saml2_authn_request.remove(header)