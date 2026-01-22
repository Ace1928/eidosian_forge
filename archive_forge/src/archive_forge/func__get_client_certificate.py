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
def _get_client_certificate(self, request):
    """Get the client certificate from request environ or socket."""
    try:
        pem_client_cert = request.environ.get('SSL_CLIENT_CERT')
        if pem_client_cert:
            peer_cert = ssl.PEM_cert_to_DER_cert(pem_client_cert)
        else:
            wsgi_input = request.environ.get('wsgi.input')
            if not wsgi_input:
                self._log.warn('Unable to obtain the client certificate. The object for wsgi_input is none.')
                raise InvalidToken(_('Unable to obtain the client certificate.'))
            socket = wsgi_input.get_socket()
            if not socket:
                self._log.warn('Unable to obtain the client certificate. The object for socket is none.')
                raise InvalidToken(_('Unable to obtain the client certificate.'))
            peer_cert = socket.getpeercert(binary_form=True)
        if not peer_cert:
            self._log.warn('Unable to obtain the client certificate. The object for peer_cert is none.')
            raise InvalidToken(_('Unable to obtain the client certificate.'))
        return peer_cert
    except InvalidToken:
        raise
    except Exception as error:
        self._log.warn('Unable to obtain the client certificate. %s' % error)
        raise InvalidToken(_('Unable to obtain the client certificate.'))