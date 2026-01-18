import taskflow.engines as engines
from taskflow.patterns import linear_flow
from taskflow import task as base
from taskflow import test
def test_square_array(self):
    expected = self.flow_store.copy()
    expected.update({'square_a': 1, 'square_b': 4, 'square_c': 9, 'square_d': 16, 'square_e': 25})
    requires = self.flow_store.keys()
    provides = ['square_%s' % k for k in requires]
    flow = linear_flow.Flow('square array flow')
    flow.add(base.MapFunctorTask(square, requires=requires, provides=provides))
    result = engines.run(flow, store=self.flow_store)
    self.assertDictEqual(expected, result)