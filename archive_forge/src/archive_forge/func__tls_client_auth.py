import flask
from flask import make_response
import http.client
from oslo_log import log
from oslo_serialization import jsonutils
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import utils
from keystone.conf import CONF
from keystone import exception
from keystone.federation import utils as federation_utils
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _tls_client_auth(self, client_id, client_cert):
    """Get an OAuth2.0 certificate-bound Access Token."""
    try:
        cert_subject_dn = utils.get_certificate_subject_dn(client_cert)
    except exception.ValidationError:
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: failed to get the subject DN from the certificate.')
        raise error
    try:
        cert_issuer_dn = utils.get_certificate_issuer_dn(client_cert)
    except exception.ValidationError:
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: failed to get the issuer DN from the certificate.')
        raise error
    client_cert_dn = {}
    for key in cert_subject_dn:
        client_cert_dn['SSL_CLIENT_SUBJECT_DN_%s' % key.upper()] = cert_subject_dn.get(key)
    for key in cert_issuer_dn:
        client_cert_dn['SSL_CLIENT_ISSUER_DN_%s' % key.upper()] = cert_issuer_dn.get(key)
    try:
        user = PROVIDERS.identity_api.get_user(client_id)
    except exception.UserNotFound:
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: the user does not exist. user id: %s.', client_id)
        raise error
    project_id = user.get('default_project_id')
    if not project_id:
        error = exception.OAuth2InvalidClient(int(http.client.UNAUTHORIZED), http.client.responses[http.client.UNAUTHORIZED], _('Client authentication failed.'))
        LOG.info('Get OAuth2.0 Access Token API: the user does not have default project. user id: %s.', client_id)
        raise error
    user_domain = PROVIDERS.resource_api.get_domain(user.get('domain_id'))
    self._check_mapped_properties(client_cert_dn, user, user_domain)
    thumbprint = utils.get_certificate_thumbprint(client_cert)
    LOG.debug(f'The mTLS certificate thumbprint: {thumbprint}')
    try:
        token = PROVIDERS.token_provider_api.issue_token(user_id=client_id, method_names=['oauth2_credential'], project_id=project_id, thumbprint=thumbprint)
    except exception.Error as error:
        if error.code == 401:
            error = exception.OAuth2InvalidClient(error.code, error.title, str(error))
        elif error.code == 400:
            error = exception.OAuth2InvalidRequest(error.code, error.title, str(error))
        else:
            error = exception.OAuth2OtherError(error.code, error.title, 'An unknown error occurred and failed to get an OAuth2.0 access token.')
        LOG.exception(error)
        raise error
    except Exception as error:
        error = exception.OAuth2OtherError(int(http.client.INTERNAL_SERVER_ERROR), http.client.responses[http.client.INTERNAL_SERVER_ERROR], str(error))
        LOG.exception(error)
        raise error
    resp = make_response({'access_token': token.id, 'token_type': 'Bearer', 'expires_in': CONF.token.expiration})
    resp.status = '200 OK'
    return resp