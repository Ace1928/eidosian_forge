import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
@contextlib.contextmanager
def Provision(allow_soft_placement=True, disable_detailed_stats=True, disable_timeline=True, devices=None):
    cluster = Cluster(allow_soft_placement, disable_detailed_stats, disable_timeline, devices)
    yield cluster
    cluster.Shutdown()