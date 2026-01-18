from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified instance group. We will retrieve the information from the GCE APIs
    every time this method is called.

    Returns:
      A ClusterSpec containing host information retrieved from GCE.
    