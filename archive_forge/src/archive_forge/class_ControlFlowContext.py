import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
class ControlFlowContext(metaclass=abc.ABCMeta):
    """The base class for control flow context.

  The usage pattern is a sequence of (Enter, Exit) followed by a final
  ExitResult.

  We maintain the following state for control flow contexts during graph
  construction:
   1. graph has _control_flow_context: the current context used to
      construct new nodes. Changed by ctxt.Enter() and ctxt.Exit()
   2. op has _control_flow_context: the context to which the op belongs.
      Set at the time the op is created. Immutable.
   3. A ControlFlowContext has _outer_context: the context in which this
      context is created. Set at the time a context is created. Immutable.
   4. A ControlFlowContext has _context_stack.
      Pushed and popped by ctxt.Enter() and ctxt.Exit()
  """

    def __init__(self, values_def=None, import_scope=None):
        self._nested_contexts = []
        self._outer_context = ops.get_default_graph()._get_control_flow_context()
        if self._outer_context:
            self._outer_context._nested_contexts.append(self)
        self._context_stack = []
        if values_def:
            self._init_values_from_proto(values_def, import_scope=import_scope)
        else:
            self._values = set()
            self._external_values = {}

    def _init_values_from_proto(self, values_def, import_scope=None):
        """Initializes values and external_values from `ValuesDef` protocol buffer.

    Args:
      values_def: `ValuesDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
        assert isinstance(values_def, control_flow_pb2.ValuesDef)
        self._values = set((ops.prepend_name_scope(value, import_scope) for value in values_def.values))
        g = ops.get_default_graph()
        self._external_values = {}
        for k, v in values_def.external_values.items():
            k = ops.prepend_name_scope(k, import_scope)
            self._external_values[k] = g.as_graph_element(ops.prepend_name_scope(v, import_scope))
        op_names = set([op.split(':')[0] for op in self._values - set(self._external_values.keys())])
        for op in op_names:
            g.as_graph_element(op)._set_control_flow_context(self)

    @property
    def name(self):
        return self._name

    @property
    def outer_context(self):
        """Return the context containing this context."""
        return self._outer_context

    @property
    def grad_state(self):
        raise NotImplementedError('Abstract method')

    @property
    def back_prop(self):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def to_control_flow_context_def(self, context_def, export_scope=None):
        """Serializes this into `context_def`.

    Args:
      context_def: a `ControlFlowContextDef` protocol buffer.
      export_scope: Optional `string`. Name scope to remove.
    """
        raise NotImplementedError('Abstract method')

    def _to_values_def(self, export_scope=None):
        """Converts the values to a `ValuesDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `ValuesDef` protocol buffer.
    """
        values_def = control_flow_pb2.ValuesDef()
        values_def.values.extend([ops.strip_name_scope(v, export_scope) for v in sorted(self._values)])
        for k, v in self._external_values.items():
            k = ops.strip_name_scope(k, export_scope)
            values_def.external_values[k] = ops.strip_name_scope(v.name, export_scope)
        return values_def

    def AddName(self, name):
        self._values.add(name)

    def Enter(self):
        """Enter this control flow context."""
        graph = ops.get_default_graph()
        self._context_stack.append(graph._get_control_flow_context())
        graph._set_control_flow_context(self)

    def Exit(self):
        """Exit this control flow context."""
        graph = ops.get_default_graph()
        last_context = self._context_stack.pop()
        graph._set_control_flow_context(last_context)

    def EnterGradientColocation(self, op, gradient_uid):
        """Start building a gradient colocated with an op."""
        if self._outer_context:
            self._outer_context.EnterGradientColocation(op, gradient_uid)

    def ExitGradientColocation(self, op, gradient_uid):
        """Start building a gradient colocated with an op."""
        if self._outer_context:
            self._outer_context.ExitGradientColocation(op, gradient_uid)

    def ExitResult(self, result):
        """Make a list of tensors available in the outer context."""
        if self._outer_context:

            def fn(x):
                self._outer_context.AddName(x.name)
                return x
            nest.map_structure(fn, result, expand_composites=True)

    def GetWhileContext(self):
        """Return the while context containing this context."""
        if self._outer_context:
            return self._outer_context.GetWhileContext()
        return None

    def _RemoveExternalControlEdges(self, op):
        """Remove any external control dependency on this op."""
        while_ctxt = self.GetWhileContext()
        if while_ctxt is None:
            internal_control_inputs, external_control_inputs = (op.control_inputs, [])
        else:
            internal_control_inputs, external_control_inputs = ([], [])
            for x in op.control_inputs:
                ctxt = util.GetOutputContext(x)
                if ctxt is not None and ctxt.GetWhileContext() == while_ctxt:
                    internal_control_inputs.append(x)
                else:
                    external_control_inputs.append(x)
        if len(internal_control_inputs) != len(op.control_inputs):
            op._remove_all_control_inputs()
            op._add_control_inputs(internal_control_inputs)
        return (internal_control_inputs, external_control_inputs)

    def AddInnerOp(self, op):
        """Notifies a scope about an operator added to an inner scope."""
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    def GetControlPivot(self):
        """Returns the pivot node for this context, or None."""
        return None

    def IsWhileContext(self):
        return False

    def IsCondContext(self):
        return False

    def IsXLAContext(self):
        return False

    def __str__(self):
        return self.name