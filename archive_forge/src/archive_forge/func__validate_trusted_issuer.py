import functools
import re
import wsgiref.util
import http.client
from keystonemiddleware import auth_token
import oslo_i18n
from oslo_log import log
from oslo_serialization import jsonutils
import webob.dec
import webob.exc
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import render_token
from keystone.common import tokenless_auth
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.models import token_model
def _validate_trusted_issuer(self, request):
    """To further filter the certificates that are trusted.

        If the config option 'trusted_issuer' is absent or does
        not contain the trusted issuer DN, no certificates
        will be allowed in tokenless authorization.

        :param env: The env contains the client issuer's attributes
        :type env: dict
        :returns: True if client_issuer is trusted; otherwise False
        """
    if not CONF.tokenless_auth.trusted_issuer:
        return False
    issuer = request.environ.get(CONF.tokenless_auth.issuer_attribute)
    if not issuer:
        msg = 'Cannot find client issuer in env by the issuer attribute - %s.'
        LOG.info(msg, CONF.tokenless_auth.issuer_attribute)
        return False
    if issuer in CONF.tokenless_auth.trusted_issuer:
        return True
    msg = 'The client issuer %(client_issuer)s does not match with the trusted issuer %(trusted_issuer)s'
    LOG.info(msg, {'client_issuer': issuer, 'trusted_issuer': CONF.tokenless_auth.trusted_issuer})
    return False