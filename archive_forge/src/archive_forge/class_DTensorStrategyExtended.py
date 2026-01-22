import functools
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
class DTensorStrategyExtended(distribute_lib.StrategyExtendedV2):
    """Strategy extension that support both single and multi worker strategy."""

    def __init__(self, container_strategy, mesh):
        super().__init__(container_strategy)
        self._mesh = mesh
        self._num_clients = d_config.num_clients()
        self._client_id = d_config.client_id()

    def _create_variable(self, next_creator, **kwargs):
        kwargs.pop('use_resource', None)
        kwargs.pop('colocate_with', None)
        kwargs.pop('expected_shape', None)
        initial_value = kwargs.pop('initial_value')
        dtype = kwargs.get('dtype', None)

        def new_initial_value():
            if callable(initial_value):
                init_var = ops.convert_to_tensor(initial_value(), dtype=dtype)
            else:
                init_var = ops.convert_to_tensor(initial_value, dtype=dtype)
            rank = init_var.shape.rank
            return d_api.copy_to_mesh(init_var, layout.Layout.replicated(self._mesh, rank))
        return d_variable.DVariable(new_initial_value, **kwargs)

    @property
    def _num_replicas_in_sync(self):
        return self._mesh.size

    def value_container(self, value):
        return value

    @property
    def worker_devices(self):
        return tuple(self._mesh.local_devices())

    @property
    def parameter_devices(self):
        return self.worker_devices

    def _in_multi_worker_mode(self):
        return d_config.num_clients() > 1

    def _get_local_replica_id(self, replica_id_in_sync_group):
        return replica_id_in_sync_group

    def _default_device_scope(self):
        return d_api.default_mesh(self._mesh)

    def _experimental_distribute_dataset(self, dataset, options):
        batch_size = distribute.compute_batch_size(dataset)
        if batch_size.numpy() < 0:
            raise ValueError('DTensor strategy requires a static batch size for now.The dynamic batch size will be supported in future')
        dataset = dataset.unbatch()

        def _create_batch_layout(tensor_spec):
            rank = len(tensor_spec.shape) + 1
            return layout.Layout.batch_sharded(self._mesh, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)
        layouts = nest.map_structure(_create_batch_layout, dataset.element_spec)
        return input_util.DTensorDataset(dataset=dataset, mesh=self._mesh, layouts=layouts, global_batch_size=batch_size, dataset_already_batched=False, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, prefetch=None, tf_data_service_config=None)

    def _make_dataset_iterator(self, dataset):
        raise NotImplementedError('Strategy.make_dataset_iterator() is deprecated, and only available in the V1 API.')

    def _make_input_fn_iterator(self, input_fn, replication_mode):
        raise NotImplementedError('Strategy.make_input_fn_iterator() is deprecated, and only available in the V1 API.')

    def _distribute_datasets_from_function(self, dataset_fn, options):
        del options
        input_context = distribute_lib.InputContext(num_input_pipelines=self._num_clients, input_pipeline_id=self._client_id, num_replicas_in_sync=self._num_replicas_in_sync)
        dataset = dataset_fn(input_context)

        def _create_batch_layout(tensor_spec):
            rank = len(tensor_spec.shape)
            return layout.Layout.batch_sharded(self._mesh, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)
        layouts = nest.map_structure(_create_batch_layout, dataset.element_spec)
        batch_size = distribute.compute_batch_size(dataset)
        if batch_size.numpy() < 0:
            raise ValueError('DTensor strategy requires a static batch size for now.The dynamic batch size will be supported in future')
        global_batch_size = batch_size.numpy() * self._num_replicas_in_sync
        return input_util.DTensorDataset(dataset=dataset, mesh=self._mesh, layouts=layouts, global_batch_size=global_batch_size, dataset_already_batched=True, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, prefetch=None, tf_data_service_config=None)

    def _experimental_distribute_values_from_function(self, value_fn):
        per_replica_values = []
        for i in range(self._mesh.num_local_devices()):
            replica_id = d_config.client_id() * self._mesh.num_local_devices() + i
            per_replica_values.append(value_fn(distribute_lib.ValueContext(replica_id, self._num_replicas_in_sync)))
        result = distribute_utils.regroup(per_replica_values, always_wrap=True)
        map_fn = functools.partial(dtensor_util.convert_per_replica_to_dtensor, mesh=self._mesh)
        return nest.map_structure(map_fn, result)

    def call_for_each_replica(self, fn, args=(), kwargs=None):
        """Run `fn` once per replica.

    This is a method that expected by the strategy base class in its `run()`.

    Args:
      fn: function to run (will be run once per replica).
      args: Tuple or list with positional arguments for `fn`.
      kwargs: Dict with keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.
    """
        distribute_lib._require_cross_replica_or_default_context_extended(self)
        if kwargs is None:
            kwargs = {}
        map_fn = functools.partial(dtensor_util.convert_inputs_to_dtensor, mesh=self._mesh)
        d_args = nest.map_structure(map_fn, args)
        d_kwargs = nest.map_structure(map_fn, kwargs)
        with self._container_strategy().scope():
            with dtensor_util.DTensorReplicaContext(self._container_strategy()):
                dtensor_result = fn(*d_args, **d_kwargs)
        return nest.map_structure(dtensor_util.DTensorDistributedValue, dtensor_result)

    def _gather_to_implementation(self, value, destinations, axis, options):
        if isinstance(value, dtensor_util.DTensorDistributedValue):
            value = value.get_dtensor()
        if not d_api.is_dtensor(value):
            return value
        components = d_api.unpack(value)
        return array_ops.concat(components, axis=axis)

    def _use_merge_call(self):
        return False