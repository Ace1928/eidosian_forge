import functools
import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.resource import schema
from keystone.server import flask as ks_flask
def _build_project_target_enforcement():
    target = {}
    try:
        target['project'] = PROVIDERS.resource_api.get_project(flask.request.view_args.get('project_id'))
    except exception.NotFound:
        pass
    return target