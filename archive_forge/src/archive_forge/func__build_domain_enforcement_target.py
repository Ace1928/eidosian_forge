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
def _build_domain_enforcement_target():
    target = {}
    try:
        target['domain'] = PROVIDERS.resource_api.get_domain(flask.request.view_args.get('domain_id'))
    except exception.NotFound:
        pass
    return target