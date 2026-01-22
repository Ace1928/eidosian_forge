import flask_restful
import http.client
from oslo_log import versionutils
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.policy import schema
from keystone.server import flask as ks_flask
@versionutils.deprecated(as_of=versionutils.deprecated.QUEENS, what='identity:list_policies of the v3 Policy APIs')
def _list_policies(self):
    ENFORCER.enforce_call(action='identity:list_policies')
    filters = ['type']
    hints = self.build_driver_hints(filters)
    refs = PROVIDERS.policy_api.list_policies(hints=hints)
    return self.wrap_collection(refs, hints=hints)