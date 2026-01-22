import sys
import unittest
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import cluster_resolver
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util as framework_test_util
from tensorflow.python.platform import flags
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export
def _create_parameter_server():
    if framework_test_util.is_xla_enabled():
        cluster_def = multi_worker_test_base.create_in_process_cluster(num_workers=num_workers, num_ps=num_ps, rpc_layer='grpc')
        resolver = cluster_resolver.SimpleClusterResolver(server_lib.ClusterSpec(cluster_def), num_accelerators={'GPU': required_gpus}, rpc_layer='grpc')
        return _create_ps_strategy(resolver, variable_partitioner)
    else:
        tf_config = cluster_resolver.TFConfigClusterResolver()
        cluster_def = tf_config.cluster_spec().as_dict()
        if not cluster_def:
            return None
        resolver = cluster_resolver.SimpleClusterResolver(server_lib.ClusterSpec(cluster_def), num_accelerators={'GPU': required_gpus}, task_type=tf_config.task_type, task_id=tf_config.task_id, environment=tf_config.environment, rpc_layer=tf_config.rpc_layer or 'grpc')
        if tf_config.task_type in ('worker', 'ps'):
            worker_config = config_pb2.ConfigProto()
            worker_config.inter_op_parallelism_threads = 4
            try:
                server = server_lib.Server(cluster_def, job_name=tf_config.task_type, task_index=tf_config.task_id, protocol='grpc', config=worker_config)
            except errors.UnknownError as e:
                if 'Could not start gRPC server' in e.message:
                    raise unittest.SkipTest('Cannot start std servers.')
                else:
                    raise
            server.join()
        return _create_ps_strategy(resolver, variable_partitioner)