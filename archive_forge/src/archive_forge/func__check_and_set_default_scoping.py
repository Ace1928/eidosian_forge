import flask
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.receipt import handlers as receipt_handlers
def _check_and_set_default_scoping(auth_info, auth_context):
    domain_id, project_id, trust, unscoped, system = auth_info.get_scope()
    if trust:
        project_id = trust['project_id']
    if system or domain_id or project_id or trust:
        return
    if constants.IDENTITY_PROVIDER in auth_context:
        return
    if unscoped is not None:
        return
    try:
        user_ref = PROVIDERS.identity_api.get_user(auth_context['user_id'])
    except exception.UserNotFound as e:
        LOG.warning(e)
        raise exception.Unauthorized(e)
    default_project_id = user_ref.get('default_project_id')
    if not default_project_id:
        return
    try:
        default_project_ref = PROVIDERS.resource_api.get_project(default_project_id)
        default_project_domain_ref = PROVIDERS.resource_api.get_domain(default_project_ref['domain_id'])
        if default_project_ref.get('enabled', True) and default_project_domain_ref.get('enabled', True):
            if PROVIDERS.assignment_api.get_roles_for_user_and_project(user_ref['id'], default_project_id):
                auth_info.set_scope(project_id=default_project_id)
            else:
                msg = "User %(user_id)s doesn't have access to default project %(project_id)s. The token will be unscoped rather than scoped to the project."
                LOG.debug(msg, {'user_id': user_ref['id'], 'project_id': default_project_id})
        else:
            msg = "User %(user_id)s's default project %(project_id)s is disabled. The token will be unscoped rather than scoped to the project."
            LOG.debug(msg, {'user_id': user_ref['id'], 'project_id': default_project_id})
    except (exception.ProjectNotFound, exception.DomainNotFound):
        msg = "User %(user_id)s's default project %(project_id)s not found. The token will be unscoped rather than scoped to the project."
        LOG.debug(msg, {'user_id': user_ref['id'], 'project_id': default_project_id})