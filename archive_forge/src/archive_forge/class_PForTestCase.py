import numpy as np
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest
class PForTestCase(test.TestCase):
    """Base class for test cases."""

    def _run_targets(self, targets1, targets2=None, run_init=True):
        targets1 = nest.flatten(targets1)
        targets2 = [] if targets2 is None else nest.flatten(targets2)
        assert len(targets1) == len(targets2) or not targets2
        if run_init:
            init = variables.global_variables_initializer()
            self.evaluate(init)
        return self.evaluate(targets1 + targets2)

    def run_and_assert_equal(self, targets1, targets2, rtol=0.0001, atol=1e-05):
        outputs = self._run_targets(targets1, targets2)
        outputs = nest.flatten(outputs)
        n = len(outputs) // 2
        for i in range(n):
            if outputs[i + n].dtype != np.object_:
                self.assertAllClose(outputs[i + n], outputs[i], rtol=rtol, atol=atol)
            else:
                self.assertAllEqual(outputs[i + n], outputs[i])

    def _test_loop_fn(self, loop_fn, iters, parallel_iterations=None, fallback_to_while_loop=False, rtol=0.0001, atol=1e-05):
        t1 = pfor_control_flow_ops.pfor(loop_fn, iters=iters, fallback_to_while_loop=fallback_to_while_loop, parallel_iterations=parallel_iterations)
        loop_fn_dtypes = nest.map_structure(lambda x: x.dtype, t1)
        t2 = pfor_control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, iters=iters, parallel_iterations=parallel_iterations)

        def _check_shape(a, b):
            msg = f'Inferred static shapes are different between two loops: {a.shape} vs {b.shape}.'
            if b.shape:
                self.assertEqual(a.shape.as_list()[0], b.shape.as_list()[0], msg)
        nest.map_structure(_check_shape, t1, t2)
        self.run_and_assert_equal(t1, t2, rtol=rtol, atol=atol)