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
def test_resource_is_not_appear_in_stack(self):
    self.stack.remove_resource(self.rsrc.name)
    func = functions.GetAtt(self.stack.defn, 'Fn::GetAtt', [self.rsrc.name, 'Foo'])
    ex = self.assertRaises(exception.InvalidTemplateReference, func.validate)
    self.assertEqual('The specified reference "test_rsrc" (in unknown) is incorrect.', str(ex))