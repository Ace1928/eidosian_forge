import taskflow.engines as engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def test_sum_array(self):
    expected = self.flow_store.copy()
    expected.update({'sum': 15})
    requires = self.flow_store.keys()
    provides = 'sum'
    flow = linear_flow.Flow('sum array flow')
    flow.add(base.ReduceFunctorTask(sum, requires=requires, provides=provides))
    result = engines.run(flow, store=self.flow_store)
    self.assertDictEqual(expected, result)