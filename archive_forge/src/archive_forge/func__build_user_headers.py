from keystoneauth1 import exceptions as keystone_exceptions
from keystoneauth1 import session
from webob import exc
from heat.common import config
from heat.common import context
def _build_user_headers(self, token_info):
    """Build headers that represent authenticated user from auth token."""
    if token_info.version == 'v3':
        project_id = token_info.project_id
        project_name = token_info.project_name
    else:
        project_id = token_info.tenant_id
        project_name = token_info.tenant_name
    user_id = token_info.user_id
    user_name = token_info.username
    roles = ','.join([role for role in token_info.role_names])
    service_catalog = token_info.service_catalog
    auth_token = token_info.auth_token
    user_domain_id = token_info.user_domain_id
    headers = {'HTTP_X_IDENTITY_STATUS': 'Confirmed', 'HTTP_X_PROJECT_ID': project_id, 'HTTP_X_PROJECT_NAME': project_name, 'HTTP_X_USER_ID': user_id, 'HTTP_X_USER_NAME': user_name, 'HTTP_X_ROLES': roles, 'HTTP_X_SERVICE_CATALOG': service_catalog, 'HTTP_X_AUTH_TOKEN': auth_token, 'HTTP_X_USER_DOMAIN_ID': user_domain_id}
    return headers