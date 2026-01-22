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
def _create_table_to_features_dict(feature_to_config_dict):
    """Create mapping from table to a list of its features."""
    table_to_features_dict_tmp = {}
    for feature, feature_config in feature_to_config_dict.items():
        if feature_config.table_id in table_to_features_dict_tmp:
            table_to_features_dict_tmp[feature_config.table_id].append(feature)
        else:
            table_to_features_dict_tmp[feature_config.table_id] = [feature]
    table_to_features_dict = collections.OrderedDict()
    for table in sorted(table_to_features_dict_tmp):
        table_to_features_dict[table] = sorted(table_to_features_dict_tmp[table])
    return table_to_features_dict