import uuid
import fixtures
from keystoneauth1 import fixture as client_fixtures
from oslo_log import log as logging
from oslo_utils import timeutils
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _exceptions
def add_token_data(self, token_id=None, expires=None, user_id=None, user_name=None, user_domain_id=None, user_domain_name=None, project_id=None, project_name=None, project_domain_id=None, project_domain_name=None, role_list=None, is_v2=False):
    """Add token data to the auth_token fixture."""
    if not token_id:
        token_id = uuid.uuid4().hex
    if not role_list:
        role_list = []
    if is_v2:
        token = client_fixtures.V2Token(token_id=token_id, expires=expires, tenant_id=project_id, tenant_name=project_name, user_id=user_id, user_name=user_name)
    else:
        token = client_fixtures.V3Token(expires=expires, user_id=user_id, user_name=user_name, user_domain_id=user_domain_id, project_id=project_id, project_name=project_name, project_domain_id=project_domain_id, user_domain_name=user_domain_name, project_domain_name=project_domain_name)
    for role in role_list:
        token.add_role(name=role)
    self.add_token(token, token_id=token_id)