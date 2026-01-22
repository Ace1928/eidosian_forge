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
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
def _build_group_target_enforcement():
    target = {}
    try:
        target['group'] = PROVIDERS.identity_api.get_group(flask.request.view_args.get('group_id'))
    except exception.NotFound:
        pass
    return target