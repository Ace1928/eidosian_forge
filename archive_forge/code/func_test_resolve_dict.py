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
def test_resolve_dict(self):
    func = TestFunction(None, 'foo', ['bar', 'baz'])
    snippet = {'foo': 'bar', 'blarg': func}
    result = function.resolve(snippet)
    self.assertEqual({'foo': 'bar', 'blarg': 'wibble'}, result)
    self.assertIsNot(result, snippet)