import collections
import copy
import os
import re
import shlex
from typing import List, Tuple
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib
def _get_variable_nodes_from_graph_def(graph_def):
    """Get the list of Variable nodes from `graph_def`.

  Args:
    graph_def: An instance of `GraphDef`.  This GraphDef *must*
      have already been optimized by Grappler.  In particular, function
      inlining must have already happened.

  Returns:
    A dict mapping string names of variables to tuples `(node_def, modified)`,
    where `node_def` is the `NodeDef` corresponding to variable, and `modified`
    is a python bool describing whether the variable is modified during runtime.
  """
    variables = [n for n in graph_def.node if n.op == 'VarHandleOp']
    variable_name_map = dict(((n.name, n) for n in variables))
    child_map = collections.defaultdict(lambda: [])
    for n in graph_def.node:
        for inp in n.input:
            if not inp.startswith('^'):
                child_map[inp].append(n)
    variables = {}
    for v_name, v_node in variable_name_map.items():
        queue = list(child_map[v_name])
        processed = set([])
        while queue:
            n_current = queue.pop()
            if n_current.name in processed:
                continue
            processed.add(n_current.name)
            if n_current.op in _PASS_THROUGH_VARIABLE_OPS:
                children = child_map.get(n_current.name, [])
                queue.extend(children)
            elif n_current.op not in _READ_ONLY_VARIABLE_OPS:
                variables[v_name] = (v_node, True)
                queue = []
        if v_name not in variables:
            variables[v_name] = (v_node, False)
    return variables