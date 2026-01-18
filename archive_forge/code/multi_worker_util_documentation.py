from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
Returns whether a worker context has been entered.