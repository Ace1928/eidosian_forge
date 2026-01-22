from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def _get_ref_args(self, node):
    """Determine whether an input of an op is ref-type.

    Args:
      node: A `NodeDef`.

    Returns:
      A list of the arg names (as strs) that are ref-type.
    """
    op_def = op_def_registry.get(node.op)
    if op_def is None:
        return []
    ref_args = []
    for i, output_arg in enumerate(op_def.output_arg):
        if output_arg.is_ref:
            arg_name = node.name if i == 0 else '%s:%d' % (node.name, i)
            ref_args.append(arg_name)
    return ref_args