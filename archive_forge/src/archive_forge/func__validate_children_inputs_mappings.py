import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
def _validate_children_inputs_mappings(self, children_inputs_mappings):
    """Validate children inputs mappings is in the right format.

    Args:
      children_inputs_mappings: the Children ophint inputs/outputs mapping.
    """
    assert isinstance(children_inputs_mappings, dict)
    assert 'parent_first_child_input' in children_inputs_mappings
    assert 'parent_last_child_output' in children_inputs_mappings
    assert 'internal_children_input_output' in children_inputs_mappings

    def assert_dictlist_has_keys(dictlist, keys):
        for dikt in dictlist:
            assert isinstance(dikt, dict)
            for key in keys:
                assert key in dikt
    assert_dictlist_has_keys(children_inputs_mappings['parent_first_child_input'], ['parent_ophint_input_index', 'first_child_ophint_input_index'])
    assert_dictlist_has_keys(children_inputs_mappings['parent_last_child_output'], ['parent_output_index', 'child_output_index'])
    assert_dictlist_has_keys(children_inputs_mappings['internal_children_input_output'], ['child_input_index', 'child_output_index'])