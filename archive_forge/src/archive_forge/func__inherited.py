import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
@property
def _inherited(self):
    inherited = None
    req_args = flask.request.args
    if 'scope.OS-INHERIT:inherited_to' in req_args:
        inherited = req_args['scope.OS-INHERIT:inherited_to'] == 'projects'
    return inherited