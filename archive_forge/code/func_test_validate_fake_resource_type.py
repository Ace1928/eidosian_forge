import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_validate_fake_resource_type(self):
    tmpl_def = copy.deepcopy(TEMPLATE)
    tmpl_res_prop = tmpl_def['resources']['test-chain']['properties']
    res_list = tmpl_res_prop['resources']
    res_list.append('foo')
    chain = self._create_chain(tmpl_def)
    try:
        chain.validate_nested_stack()
        self.fail('Exception expected')
    except exception.StackValidationFailed as e:
        self.assertIn('could not be found', e.message.lower())
        self.assertIn('foo', e.message)