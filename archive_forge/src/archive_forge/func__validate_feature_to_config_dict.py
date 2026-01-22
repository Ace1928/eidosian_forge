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
def _validate_feature_to_config_dict(table_to_config_dict, feature_to_config_dict):
    """Validate `feature_to_config_dict`."""
    used_table_set = set([feature.table_id for feature in feature_to_config_dict.values()])
    table_set = set(table_to_config_dict.keys())
    unused_table_set = table_set - used_table_set
    if unused_table_set:
        raise ValueError('`table_to_config_dict` specifies table that is not used in `feature_to_config_dict`: {}.'.format(unused_table_set))
    extra_table_set = used_table_set - table_set
    if extra_table_set:
        raise ValueError('`feature_to_config_dict` refers to a table that is not specified in `table_to_config_dict`: {}.'.format(extra_table_set))