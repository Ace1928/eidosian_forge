import flask
from oslo_log import log
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def token_authenticate(token):
    response_data = {}
    try:
        json_body = flask.request.get_json(silent=True, force=True) or {}
        project_scoped = 'project' in json_body['auth'].get('scope', {})
        domain_scoped = 'domain' in json_body['auth'].get('scope', {})
        if token.oauth_scoped:
            raise exception.ForbiddenAction(action=_('Using OAuth-scoped token to create another token. Create a new OAuth-scoped token instead'))
        elif token.trust_scoped:
            raise exception.ForbiddenAction(action=_('Using trust-scoped token to create another token. Create a new trust-scoped token instead'))
        elif token.system_scoped and (project_scoped or domain_scoped):
            raise exception.ForbiddenAction(action=_('Using a system-scoped token to create a project-scoped or domain-scoped token is not allowed.'))
        if not CONF.token.allow_rescope_scoped_token:
            if token.project_scoped or token.domain_scoped:
                raise exception.ForbiddenAction(action=_('rescope a scoped token'))
        try:
            token_audit_id = token.parent_audit_id or token.audit_id
        except IndexError:
            token_audit_id = None
        response_data.setdefault('expires_at', token.expires_at)
        response_data['audit_id'] = token_audit_id
        response_data.setdefault('user_id', token.user_id)
        return response_data
    except AssertionError as e:
        LOG.error(e)
        raise exception.Unauthorized(e)