from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_serialization import jsonutils
def _subscription_options(self):
    params = dict(self.properties[self.PARAMS])
    params.setdefault('env', {})
    params['env']['notification'] = '$zaqar_message$'
    post_data = {self.WORKFLOW_ID: self.properties[self.WORKFLOW_ID], self.PARAMS: params, self.INPUT: self.properties[self.INPUT]}
    return {'post_data': jsonutils.dumps(post_data)}