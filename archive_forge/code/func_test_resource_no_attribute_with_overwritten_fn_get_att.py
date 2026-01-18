import copy
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions
from heat.engine import environment
from heat.engine import function
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resource_no_attribute_with_overwritten_fn_get_att(self):
    res_defn = rsrc_defn.ResourceDefinition('test_rsrc', 'OS::Test::FakeResource')
    self.rsrc = resource.Resource('test_rsrc', res_defn, self.stack)
    self.rsrc.attributes_schema = {}
    self.stack.add_resource(self.rsrc)
    stk_defn.update_resource_data(self.stack.defn, self.rsrc.name, self.rsrc.node_data())
    self.stack.validate()
    func = functions.GetAtt(self.stack.defn, 'Fn::GetAtt', [self.rsrc.name, 'Foo'])
    self.assertIsNone(func.validate())