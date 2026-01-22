import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
class TestBase(test_base.DatasetTestBase):
    """Base class for tf.data service tests."""

    def setUp(self):
        self.default_data_transfer_protocol = None
        self.default_compression = 'AUTO'

    def set_default_data_transfer_protocol(self, protocol):
        self.default_data_transfer_protocol = protocol

    def set_default_compression(self, compression):
        self.default_compression = compression

    def make_test_cluster(self, *args, **kwargs):
        if 'data_transfer_protocol' not in kwargs:
            kwargs['data_transfer_protocol'] = self.default_data_transfer_protocol
        return TestCluster(*args, **kwargs)

    def make_distributed_dataset(self, dataset, cluster, processing_mode='parallel_epochs', **kwargs):
        kwargs['task_refresh_interval_hint_ms'] = 20
        if 'data_transfer_protocol' not in kwargs:
            kwargs['data_transfer_protocol'] = self.default_data_transfer_protocol
        if 'compression' not in kwargs:
            kwargs['compression'] = self.default_compression
        return dataset.apply(data_service_ops._distribute(processing_mode, cluster.dispatcher_address(), **kwargs))

    def make_distributed_range_dataset(self, num_elements, cluster, **kwargs):
        dataset = dataset_ops.Dataset.range(num_elements)
        return self.make_distributed_dataset(dataset, cluster, **kwargs)

    def make_coordinated_read_dataset(self, cluster, num_consumers, sharding_policy=data_service_ops.ShardingPolicy.OFF):
        """Creates a dataset that performs coordinated reads.

    The dataset simulates `num_consumers` consumers by using parallel
    interleave to read with `num_consumers` threads, one for each consumer. The
    nth element of the dataset is produced by consumer `n % num_consumers`.

    The dataset executed on each worker will produce groups of `num_consumers`
    sequentially increasing numbers. For example, if `num_consumers=3` a worker
    dataset could produce [0, 1, 2, 9, 10, 11, 21, 22, 23]. This enables
    `checkCoordinatedReadGroups` below to assess whether the values received in
    each step came from the same group.

    Args:
      cluster: A tf.data service `TestCluster`.
      num_consumers: The number of consumers to simulate.
      sharding_policy: The sharding policy to use. Currently only OFF and
        DYNAMIC are supported.

    Returns:
      A dataset that simulates reading with `num_consumers` consumers.
    """
        if sharding_policy not in [data_service_ops.ShardingPolicy.OFF, data_service_ops.ShardingPolicy.DYNAMIC]:
            raise ValueError(f'Unsupported sharding policy: {sharding_policy}')
        ds = dataset_ops.Dataset.from_tensors(math_ops.cast(0, dtypes.int64))
        ds = ds.concatenate(dataset_ops.Dataset.random())

        def make_group(x):
            x = x % 2 ** 32
            return dataset_ops.Dataset.range(x * num_consumers, (x + 1) * num_consumers)
        ds = ds.flat_map(make_group)
        consumers = []
        for consumer_index in range(num_consumers):
            consumers.append(self.make_distributed_dataset(ds, cluster, job_name='test', processing_mode=sharding_policy, consumer_index=consumer_index, num_consumers=num_consumers))
        ds = dataset_ops.Dataset.from_tensor_slices(consumers)
        ds = ds.interleave(lambda x: x, cycle_length=num_consumers, num_parallel_calls=num_consumers)
        return ds

    def checkCoordinatedReadGroups(self, results, num_consumers):
        """Validates results from a `make_coordinted_read_dataset` dataset.

    Each group of `num_consumers` results should be consecutive, indicating that
    they were produced by the same worker.

    Args:
      results: The elements produced by the dataset.
      num_consumers: The number of consumers.
    """
        groups = [results[start:start + num_consumers] for start in range(0, len(results), num_consumers)]
        incorrect_groups = []
        for group in groups:
            for offset in range(1, len(group)):
                if group[0] + offset != group[offset]:
                    incorrect_groups.append(group)
                    break
        self.assertEmpty(incorrect_groups, 'Incorrect groups: {}.\nAll groups: {}'.format(incorrect_groups, groups))

    def read(self, get_next, results, count):
        for _ in range(count):
            results.append(self.evaluate(get_next()))