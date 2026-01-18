from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import scopes as sc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import test
from taskflow.tests import utils as test_utils
def test_shadow_linear(self):
    r = lf.Flow('root')
    customer = test_utils.ProvidesRequiresTask('customer', provides=['dog'], requires=[])
    customer2 = test_utils.ProvidesRequiresTask('customer2', provides=['dog'], requires=[])
    washer = test_utils.ProvidesRequiresTask('washer', requires=['dog'], provides=['wash'])
    r.add(customer, customer2, washer)
    c = compiler.PatternCompiler(r).compile()
    self.assertEqual(['customer2', 'customer'], _get_scopes(c, washer)[0])