import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_policy import _checks as op_checks
from keystone.api._shared import json_home_relations
from keystone.common import context
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
from keystone.trust import schema
def _require_trustor_has_role_in_project(self, trust):
    trustor_roles = self._get_trustor_roles(trust)
    for trust_role in trust['roles']:
        matching_roles = [x for x in trustor_roles if x == trust_role['id']]
        if not matching_roles:
            raise exception.RoleNotFound(role_id=trust_role['id'])