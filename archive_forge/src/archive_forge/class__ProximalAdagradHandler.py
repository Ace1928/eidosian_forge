import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
class _ProximalAdagradHandler(_OptimizerHandler):
    """Handles ProximalAdagrad specific logic."""

    def set_optimization_parameters(self, table_descriptor):
        table_descriptor.optimization_parameters.proximal_adagrad.SetInParent()
        table_descriptor.optimization_parameters.proximal_adagrad.l1 = self._optimization_parameters.l1_regularization_strength
        table_descriptor.optimization_parameters.proximal_adagrad.l2 = self._optimization_parameters.l2_regularization_strength

    def get_default_slot_variable_names(self, table):
        return ProximalAdagradSlotVariableNames('{}/{}'.format(table, 'ProximalAdagrad'))

    def create_variables_and_ops(self, table, slot_variable_names, num_hosts, table_config, table_variables, config_proto):
        accumulator_initializer = init_ops.constant_initializer(self._optimization_parameters.initial_accumulator)
        accumulator_variables = _create_partitioned_variables(name=slot_variable_names.accumulator, num_hosts=num_hosts, vocabulary_size=table_config.vocabulary_size, embedding_dimension=table_config.dimension, collections=[ops.GraphKeys.GLOBAL_VARIABLES], initializer=accumulator_initializer)
        slot_variables = ProximalAdagradSlotVariables(accumulator_variables)

        def load_ops_fn():
            """Returns the retrieve ops for Proximal AdaGrad embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
            config = config_proto
            load_op_list = []
            for host_id, table_variable, accumulator_variable in zip(range(num_hosts), table_variables, accumulator_variables):
                with ops.colocate_with(table_variable):
                    load_parameters_op = tpu_ops.load_tpu_embedding_proximal_adagrad_parameters(parameters=table_variable, accumulators=accumulator_variable, table_name=table, num_shards=num_hosts, shard_id=host_id, config=config)
                config = None
                load_op_list.append(load_parameters_op)
            return load_op_list

        def retrieve_ops_fn():
            """Returns the retrieve ops for Proximal AdaGrad embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
            config = config_proto
            retrieve_op_list = []
            for host_id, table_variable, accumulator_variable in zip(range(num_hosts), table_variables, accumulator_variables):
                with ops.colocate_with(table_variable):
                    retrieved_table, retrieved_accumulator = tpu_ops.retrieve_tpu_embedding_proximal_adagrad_parameters(table_name=table, num_shards=num_hosts, shard_id=host_id, config=config)
                    retrieve_parameters_op = control_flow_ops.group(state_ops.assign(table_variable, retrieved_table), state_ops.assign(accumulator_variable, retrieved_accumulator))
                config = None
                retrieve_op_list.append(retrieve_parameters_op)
            return retrieve_op_list
        return (slot_variables, load_ops_fn, retrieve_ops_fn)