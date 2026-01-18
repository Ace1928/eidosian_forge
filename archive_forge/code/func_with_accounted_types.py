import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_accounted_types(self, account_type_regexes):
    """Selectively counting statistics based on node types.

    Here, 'types' means the profiler nodes' properties. Profiler by default
    consider device name (e.g. /job:xx/.../device:GPU:0) and operation type
    (e.g. MatMul) as profiler nodes' properties. User can also associate
    customized 'types' to profiler nodes through OpLogProto proto.

    For example, user can select profiler nodes placed on gpu:0 with:
    `account_type_regexes=['.*gpu:0.*']`

    If none of a node's properties match the specified regexes, the node is
    not displayed nor accounted.

    Args:
      account_type_regexes: A list of regexes specifying the types.
    Returns:
      self.
    """
    self._options['account_type_regexes'] = copy.copy(account_type_regexes)
    return self