import json
import os
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _get_value_in_tfconfig(key, port, default=None):
    tf_config = _load_tf_config(port)
    return tf_config[key] if key in tf_config else default