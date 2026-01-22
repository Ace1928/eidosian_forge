import contextlib
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.grappler import _pywrap_tf_cluster as tf_cluster
def Shutdown(self):
    if self._tf_cluster is not None:
        tf_cluster.TF_ShutdownCluster(self._tf_cluster)
        self._tf_cluster = None