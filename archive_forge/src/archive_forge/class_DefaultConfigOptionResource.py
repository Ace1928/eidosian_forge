import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.resource import schema
from keystone.server import flask as ks_flask
class DefaultConfigOptionResource(flask_restful.Resource):

    def get(self, group=None, option=None):
        """Get default domain group option config.

        GET/HEAD /v3/domains/config/{group}/{option}/default
        """
        ENFORCER.enforce_call(action='identity:get_domain_config_default')
        ref = PROVIDERS.domain_config_api.get_config_default(group=group, option=option)
        return {'config': ref}