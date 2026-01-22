import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
def MeasureCosts(self, item):
    """Returns the cost of running the specified item.

    Args:
      item: The item for which to measure the costs.
    Returns: The triplet op_perfs, runtime, step_stats.
    """
    op_perf_bytes_list, run_time, step_stats_bytes = tf_cluster.TF_MeasureCosts(item.tf_item, self._tf_cluster, self._generate_timeline)
    op_perfs = [op_performance_data_pb2.OpPerformance.FromString(op_perf_bytes) for op_perf_bytes in op_perf_bytes_list]
    return (op_perfs, run_time, step_stats_pb2.StepStats.FromString(step_stats_bytes))