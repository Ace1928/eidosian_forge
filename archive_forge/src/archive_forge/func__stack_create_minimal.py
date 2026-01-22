from tempest.lib.common.utils import data_utils as utils
from heatclient.tests.functional import config
from heatclient.tests.functional.osc.v1 import base
def _stack_create_minimal(self, from_url=False):
    if from_url:
        template = config.HEAT_MINIMAL_HOT_TEMPLATE_URL
    else:
        template = self.get_template_path('heat_minimal_hot.yaml')
    parameters = ['test_client_name=test_client_name']
    return self._stack_create(self.stack_name, template=template, parameters=parameters)