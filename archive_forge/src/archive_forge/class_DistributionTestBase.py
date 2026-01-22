import functools
import os
import tempfile
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
class DistributionTestBase(test.TestCase):
    """Some tests that should work with any DistributionStrategy."""

    def _test_minimize_loss_eager(self, d):
        with d.scope():
            kernel = create_variable_like_keras_layer(name='kernel', shape=(1, 1), dtype=dtypes.float32)

            def loss(x):
                y = array_ops.reshape(math_ops.mat_mul(x, kernel), []) - array_ops.identity(1.0)
                return y * y
            grad_fn = backprop.implicit_grad(loss)
            grad_fn = optimizer.get_filtered_grad_fn(grad_fn)

            def update(v, g):
                return v.assign_sub(0.2 * g)
            one = array_ops.identity([[1.0]])

            def step():
                """Perform one optimization step."""
                g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))
                before_list = []
                after_list = []
                for g, v in g_v:
                    fetched = d.extended.read_var(v)
                    before_list.append(fetched)
                    with ops.control_dependencies([fetched]):
                        g = d.extended.reduce_to(reduce_util.ReduceOp.SUM, g, destinations=v)
                        with ops.control_dependencies(d.extended.update(v, update, args=(g,), group=False)):
                            after_list.append(d.extended.read_var(v))
                return (before_list, after_list)
            for i in range(10):
                b, a = step()
                if i == 0:
                    before, = b
                after, = a
            error_before = abs(before.numpy() - 1)
            error_after = abs(after.numpy() - 1)
            self.assertLess(error_after, error_before)

    def _test_minimize_loss_graph(self, d, soft_placement=False, learning_rate=0.2):
        config = config_pb2.ConfigProto()
        config.allow_soft_placement = soft_placement
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with context.graph_mode(), ops.Graph().as_default(), self.cached_session(config=config) as sess, d.scope():
            kernel = create_variable_like_keras_layer(name='kernel', shape=(1, 1), dtype=dtypes.float32)

            def loss(x):
                y = array_ops.reshape(math_ops.mat_mul(x, kernel), []) - array_ops.identity(1.0)
                return y * y
            grad_fn = backprop.implicit_grad(loss)

            def update(v, g):
                return v.assign_sub(learning_rate * g)
            one = array_ops.identity([[1.0]])

            def step():
                """Perform one optimization step."""
                g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))
                before_list = []
                after_list = []
                for g, v in g_v:
                    fetched = d.extended.read_var(v)
                    before_list.append(fetched)
                    with ops.control_dependencies([fetched]):
                        g = d.extended.reduce_to(reduce_util.ReduceOp.SUM, g, destinations=v)
                        with ops.control_dependencies(d.extended.update(v, update, args=(g,), group=False)):
                            after_list.append(d.extended.read_var(v))
                return (before_list, after_list)
            before_out, after_out = step()
            variables.global_variables_initializer().run()
            for i in range(10):
                b, a = sess.run((before_out, after_out))
                if i == 0:
                    before, = b
                after, = a
            error_before = abs(before - 1)
            error_after = abs(after - 1)
            self.assertLess(error_after, error_before)

    def _test_summary_for_replica_zero_only(self, d):
        logdir = tempfile.mkdtemp()

        def run_fn():
            """Function executed for each replica."""
            with summary_writer.as_default():
                replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
                return summary_ops.write('a', replica_id)
        with self.cached_session() as sess, d.scope(), summary_ops.always_record_summaries():
            global_step = training_util.get_or_create_global_step()
            if not context.executing_eagerly():
                global_step.initializer.run()
            summary_ops.set_step(0)
            summary_writer = summary_ops.create_file_writer(logdir)
            output = d.extended.call_for_each_replica(run_fn)
            unwrapped = d.unwrap(output)
            if not context.executing_eagerly():
                sess.run(summary_writer.init())
                sess.run(unwrapped)
                sess.run(summary_writer.close())
            events = _events_from_logdir(self, logdir)
            self.assertLen(events, 2)
            self.assertEqual(events[1].summary.value[0].tag, 'a')
            self.assertEqual(events[1].summary.value[0].simple_value, 0.0)

    def _test_replica_id(self, d):
        with d.scope():
            expected_devices = [False] * len(d.extended.worker_devices)

            def mark_devices_fn():
                replica_id = self.evaluate(distribute_lib.get_replica_context().replica_id_in_sync_group)
                self.assertLess(replica_id, len(d.extended.worker_devices))
                self.assertFalse(expected_devices[replica_id])
                expected_devices[replica_id] = True
            d.extended.call_for_each_replica(mark_devices_fn)
            self.assertAllEqual(expected_devices, [True] * len(d.extended.worker_devices))

    def _test_call_and_merge_exceptions(self, dist):
        with dist.scope():
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_raise_exception_fn)
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_merge_raises_fn)
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_merge_call_raises_fn)
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_merge_call_merge_raises_fn)

    def _input_fn_to_test_input_context(self, dataset_or_callable_fn, expected_num_replicas_in_sync, expected_num_input_pipelines, expected_input_pipeline_id):
        worker_id_counter = [0]

        def _input_fn(input_context):
            """Input fn for testing."""
            self.assertIsNotNone(input_context)
            self.assertEqual(expected_num_replicas_in_sync, input_context.num_replicas_in_sync)
            self.assertEqual(expected_num_input_pipelines, input_context.num_input_pipelines)
            if expected_input_pipeline_id is not None:
                self.assertEqual(expected_input_pipeline_id, input_context.input_pipeline_id)
            else:
                self.assertEqual(worker_id_counter[0], input_context.input_pipeline_id)
                worker_id_counter[0] += 1
            return dataset_or_callable_fn()
        return _input_fn

    def _test_input_fn_iterable(self, strategy, input_fn, expected_values, ignore_order=False):
        assert_same = self.assertCountEqual if ignore_order else self.assertEqual
        iterable = strategy.distribute_datasets_from_function(input_fn)
        if context.executing_eagerly():
            iterator = iter(iterable)
            for expected_value in expected_values:
                computed_value = self.evaluate(list(strategy.experimental_local_results(next(iterator))))
                assert_same(expected_value, computed_value)
            with self.assertRaises(StopIteration):
                self.evaluate(strategy.experimental_local_results(next(iterator)))
            iterator = iter(iterable)
            for expected_value in expected_values:
                computed_value = self.evaluate(list(strategy.experimental_local_results(next(iterator))))
                assert_same(expected_value, computed_value)
        else:
            iterator = dataset_ops.make_initializable_iterator(iterable)
            self._test_input_fn_iterator(iterator, strategy.extended.worker_devices, expected_values, test_reinitialize=True, ignore_order=ignore_order)

    def _test_input_fn_iterator(self, iterator, devices, expected_values, sess=None, test_reinitialize=True, ignore_order=False):
        evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
        evaluate(iterator.initializer)
        for expected_value in expected_values:
            next_element = iterator.get_next()
            computed_value = evaluate([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
            if ignore_order:
                self.assertCountEqual(expected_value, computed_value)
            else:
                self.assertEqual(expected_value, computed_value)
        with self.assertRaises(errors.OutOfRangeError):
            next_element = iterator.get_next()
            evaluate([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
        if test_reinitialize:
            evaluate(iterator.initializer)
            for expected_value in expected_values:
                next_element = iterator.get_next()
                computed_value = evaluate([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
                if ignore_order:
                    self.assertCountEqual(expected_value, computed_value)
                else:
                    self.assertEqual(expected_value, computed_value)

    def _test_global_step_update(self, strategy):
        with strategy.scope():
            global_step = variable_scope.get_variable('global_step', shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
            self.evaluate(variables.global_variables_initializer())

            def model_fn():
                train_op = global_step.assign_add(1)
                value = global_step.read_value()
                return (train_op, value)
            train_ops, value = strategy.extended.call_for_each_replica(model_fn)
            self.evaluate(strategy.group(train_ops))
            global_step_tensors = strategy.experimental_local_results(value)
            global_step_values = self.evaluate(global_step_tensors)
            self.assertEqual((1,) * len(global_step_tensors), global_step_values)

    def _test_numpy_dataset(self, strategy, session=None, run_in_function=False):
        if not isinstance(strategy, distribute_lib.StrategyV1):
            self.skipTest('n/a: V1 only')
        cached_session = session or self.cached_session()
        with strategy.scope(), cached_session as sess:
            x = np.asarray([[1, 2], [6, 12], [2, 4], [5, 10], [3, 6], [4, 8]])
            y = np.asarray([5, 4, 3, 2, 1, 0])
            batch_size = 6
            if not strategy.extended._global_batch_size:
                batch_size = batch_size // strategy.num_replicas_in_sync
            ds = strategy.extended.experimental_make_numpy_dataset((x, y), session=sess or self.cached_session())
            ds = ds.repeat(2)
            drop_remainder = strategy.extended.experimental_require_static_shapes
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)
            i = strategy.make_dataset_iterator(ds)
            self.evaluate(i.initializer)

            def run_and_concatenate(strategy, i):
                x, y = strategy.experimental_run(_maybe_run_in_function(lambda z: z, run_in_function), i)
                x, y = self.evaluate((strategy.experimental_local_results(x), strategy.experimental_local_results(y)))
                return (np.concatenate(x), np.concatenate(y))
            x_1, y_1 = run_and_concatenate(strategy, i)
            self.assertAllEqual(x, x_1)
            self.assertAllEqual(y, y_1)
            x_2, y_2 = run_and_concatenate(strategy, i)
            self.assertAllEqual(x, x_2)
            self.assertAllEqual(y, y_2)
            with self.assertRaises(errors.OutOfRangeError):
                run_and_concatenate(strategy, i)

    def _test_trainable_variable(self, strategy):
        for cls in [variable_v1.VariableV1, variables.Variable]:
            with strategy.scope():
                v1 = cls(1.0)
                self.assertEqual(True, v1.trainable)
                v2 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ)
                self.assertEqual(False, v2.trainable)
                v3 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ, trainable=True)
                self.assertEqual(True, v3.trainable)
                v4 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ, trainable=False)
                self.assertEqual(False, v4.trainable)