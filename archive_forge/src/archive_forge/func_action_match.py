import routes
import webob
from heat.api.cfn.v1 import signal
from heat.api.cfn.v1 import stacks
from heat.common import wsgi
def action_match(environ, result):
    req = webob.Request(environ)
    env_action = req.params.get('Action')
    return env_action == api_action