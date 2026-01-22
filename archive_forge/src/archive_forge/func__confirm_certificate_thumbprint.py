import abc
import copy
import hashlib
import os
import ssl
import time
import uuid
import jwt.utils
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import requests.auth
import webob.dec
import webob.exc
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.exceptions import ConfigurationError
from keystonemiddleware.exceptions import KeystoneMiddlewareException
from keystonemiddleware.i18n import _
def _confirm_certificate_thumbprint(self, request, origin_token_metadata):
    """Check if the thumbprint in the token is valid."""
    peer_cert = self._get_client_certificate(request)
    try:
        thumb_sha256 = hashlib.sha256(peer_cert).digest()
        cert_thumb = jwt.utils.base64url_encode(thumb_sha256).decode('ascii')
    except Exception as error:
        self._log.warn('An Exception occurred. %s' % error)
        raise InvalidToken(_('Can not generate the thumbprint.'))
    token_thumb = origin_token_metadata.get('cnf', {}).get('x5t#S256')
    if cert_thumb != token_thumb:
        self._log.warn('The two thumbprints do not match. token_thumbprint: %s, certificate_thumbprint %s' % (token_thumb, cert_thumb))
        raise InvalidToken(_('The two thumbprints do not match.'))