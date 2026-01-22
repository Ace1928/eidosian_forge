import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
class WhileOp:
    """Object for storing state for converting the outputs of a while_loop."""

    def __init__(self, exit_node, pfor_ops, fallback_to_while_loop, pfor_config):
        """Initializer.

    Args:
      exit_node: A tensor output from the while_loop.
      pfor_ops: list of ops inside the current pfor loop.
      fallback_to_while_loop: If True, fallback to while loop when conversion of
        an op is not supported
      pfor_config: PForConfig object used while constructing loop body.
    """
        self._fallback_to_while_loop = fallback_to_while_loop
        self._pfor_config = pfor_config
        self._pfor_ops = set(pfor_ops)
        self._pfor_op_ids = set((x._id for x in pfor_ops))
        assert isinstance(exit_node, tensor_lib.Tensor)
        self._while_context = exit_node.op._get_control_flow_context()
        assert isinstance(self._while_context, control_flow_ops.WhileContext)
        self._context_name = self._while_context.name
        self._condition = self._while_context.pivot.op.inputs[0]
        self._is_inside_loop = self.op_is_inside_loop(self._condition.op)
        if self._is_inside_loop:
            for e in self._while_context.loop_exits:
                assert self.op_is_inside_loop(e.op)
        self._exit_switches = []
        self._body_outputs = []
        self._next_iter_control_inputs = []
        self._enter_merges = []
        self._outputs = []
        self._enters = []
        self._direct_enters = []
        for e in self._while_context.loop_exits:
            self._outputs.append(e.op.outputs[0])
            switch = e.op.inputs[0].op
            assert switch.type == 'Switch', switch
            self._exit_switches.append(switch)
            merge = switch.inputs[0].op
            assert merge.type == 'Merge', merge
            self._enter_merges.append(merge)
            enter = merge.inputs[0].op
            assert enter.type == 'Enter', enter
            self._enters.append(enter.outputs[0])
            next_iter = merge.inputs[1].op
            assert next_iter.type == 'NextIteration', next_iter
            self._body_outputs.append(next_iter.inputs[0])
            self._next_iter_control_inputs.append(next_iter.control_inputs)
        self._is_stateful = False
        for op in ops.get_default_graph().get_operations():
            control_flow_context = op._get_control_flow_context()
            if control_flow_context is None:
                continue
            if control_flow_context.name == self._context_name:
                self._is_stateful |= _is_stateful_pfor_op(op)
                if op.type == 'Enter':
                    output = op.outputs[0]
                    if output not in self._enters:
                        if output.dtype in (dtypes.resource, dtypes.variant):
                            if output not in self._direct_enters:
                                self._direct_enters.append(output)
                        else:
                            self._enters.append(output)

    def __str__(self):
        """String representation."""
        return 'while_loop(%s)' % self.name

    @property
    def inputs(self):
        """Input to all the Enter nodes."""
        return [x.op.inputs[0] for x in self._enters + self._direct_enters]

    @property
    def control_inputs(self):
        """Control input to all the Enter nodes."""
        control_inputs = []
        for x in self._enters + self._direct_enters:
            control_inputs.extend(x.op.control_inputs)
        return control_inputs

    @property
    def outputs(self):
        """Outputs of all the Exit nodes."""
        return self._outputs

    @property
    def name(self):
        """Context name for the while loop."""
        return self._context_name

    @property
    def is_inside_loop(self):
        """Returns true if the while_loop was created inside the pfor."""
        return self._is_inside_loop

    def op_is_inside_loop(self, op):
        """True if op was created inside the pfor loop body."""
        assert isinstance(op, ops.Operation)
        return op._id in self._pfor_op_ids

    @property
    def is_stateful(self):
        return self._is_stateful

    @property
    def pfor_converter(self):
        """Return a converter for the while loop."""
        return self

    def _init_pfor(self, parent_pfor, indices, cond_stacked, inputs, inputs_stacked):
        """Create a PFor object for converting parts of the while_loop.

    Args:
      parent_pfor: PFor object being used for converting the while_loop.
      indices: int32 Tensor of ids for the iterations that are still active
        (i.e. did not exit the while_loop).
      cond_stacked: True if the while_loop condition is stacked.
      inputs: list of input Tensors corresponding 1-to-1 with self._enters. Note
        that these Tensors are a subset of the loop variables for the generated
        while_loop.
      inputs_stacked: List of booleans corresponding 1-to-1 with `inputs`,
        indicating if the value is stacked or not.

    Returns:
      A PFor instance. The instance is initialized by adding conversion mappings
        of nodes that will be external to the conversion that the returned
        instance will be used for. e.g. Enter nodes as well as Merge and Switch
        outputs are mapped to converted values.
    """
        num_outputs = len(self._outputs)
        assert len(inputs) == len(self._enters)
        assert len(inputs_stacked) == len(self._enters)
        loop_var = parent_pfor.loop_var
        loop_len = array_ops.size(indices)
        pfor = PFor(loop_var, loop_len, pfor_ops=self._pfor_ops, all_indices=indices, all_indices_partitioned=cond_stacked, fallback_to_while_loop=self._fallback_to_while_loop, pfor_config=self._pfor_config)
        for enter in self._direct_enters:
            enter_input = enter.op.inputs[0]
            converted_enter, stacked, is_sparse_stacked = parent_pfor._convert_helper(enter_input)
            assert not stacked and (not is_sparse_stacked), (enter, converted_enter)
            pfor._add_conversion(enter, wrap(converted_enter, False))
        for enter, inp, stacked in zip(self._enters, inputs, inputs_stacked):
            pfor._add_conversion(enter, wrap(inp, stacked))
        for i in range(num_outputs):
            wrapped_inp = wrap(inputs[i], inputs_stacked[i])
            merge = self._enter_merges[i]
            pfor._add_conversion(merge.outputs[0], wrapped_inp)
            pfor._add_conversion(merge.outputs[1], wrap(constant_op.constant(-1.0), False))
            switch = self._exit_switches[i]
            pfor._add_conversion(switch.outputs[1], wrapped_inp)
        return pfor

    def _convert_enter(self, parent_pfor, enter):
        """Converts an Enter node."""
        inp, stacked, _ = parent_pfor._convert_helper(enter.op.inputs[0])
        control_inputs = []
        for x in enter.op.control_inputs:
            converted = parent_pfor._convert_helper(x)
            if not isinstance(converted, ops.Operation):
                converted = converted.t
            control_inputs.append(converted)
        if control_inputs:
            with ops.control_dependencies(control_inputs):
                inp = array_ops.identity(inp)
        return (inp, stacked)

    def _maybe_stacked(self, cache, inp):
        """Heuristic to figure out if the converting inp leads to a stacked value.


    Args:
      cache: map from Tensor to boolean indicating stacked/unstacked.
      inp: input Tensor.

    Returns:
      True if `inp` could get stacked. If the function returns False, the
      converted value should be guaranteed to be unstacked. If returning True,
      it may or may not be stacked.
    """
        if inp in cache:
            return cache[inp]
        if not self.op_is_inside_loop(inp.op):
            return False
        op = inp.op
        output = False
        if op.type in ['Shape', 'Rank', 'ShapeN', 'ZerosLike', 'TensorArrayV3', 'TensorArraySizeV3']:
            output = False
        elif _is_stateful_pfor_op(op):
            output = True
        elif op.type == 'Exit':
            output = True
        else:
            for t in op.inputs:
                if self._maybe_stacked(cache, t):
                    output = True
                    break
        cache[inp] = output
        return output

    def _create_init_values(self, pfor_input):
        """Create arguments passed to converted while_loop."""
        with ops.name_scope('while_init'):
            loop_len_vector = pfor_input.pfor.loop_len_vector
            loop_len = loop_len_vector[0]
            num_outputs = len(self._outputs)
            inputs = []
            maybe_stacked_cache = {}
            for i, enter in enumerate(self._enters):
                inp, stacked = self._convert_enter(pfor_input.pfor, enter)
                inputs.append(inp)
                maybe_stacked_cache[enter] = stacked
                if i < num_outputs:
                    maybe_stacked_cache[self._exit_switches[i].outputs[1]] = stacked
            input_shape_invariants = []
            output_tas = []
            ta_shape_invariants = []
            inputs_stacked = []
            for i, inp in enumerate(inputs):
                enter = self._enters[i]
                inp_stacked = self._maybe_stacked(maybe_stacked_cache, enter)
                if i < num_outputs:
                    body_output = self._body_outputs[i]
                    if enter.op in self._pfor_ops:
                        body_output_stacked = self._maybe_stacked(maybe_stacked_cache, body_output)
                    else:
                        body_output_stacked = False
                    if body_output_stacked and (not inp_stacked):
                        inp = _stack(inp, loop_len_vector).t
                        inputs[i] = inp
                        inp_stacked = True
                    output_tas.append(tensor_array_ops.TensorArray(inp.dtype, loop_len))
                    ta_shape_invariants.append(tensor_shape.TensorShape(None))
                inputs_stacked.append(inp_stacked)
                input_shape_invariants.append(tensor_shape.TensorShape(None))
            init_values = [True, pfor_input.pfor.all_indices] + inputs + output_tas
            shape_invariants = [tensor_shape.TensorShape(None), tensor_shape.TensorShape(None)] + input_shape_invariants + ta_shape_invariants
            return (init_values, inputs_stacked, shape_invariants)

    def _process_cond_unstacked(self, conditions, indices, inputs, output_tas):
        """Handles case when condition is unstacked.

    Note that all iterations end together. So we don't need to partition the
    inputs. When all iterations are done, we write the inputs to the
    TensorArrays. Note that we only write to index 0 of output_tas. Since all
    iterations end together, they can all be output together.
    """
        not_all_done = array_ops.reshape(conditions, [])
        new_output_tas = []
        for i, out_ta in enumerate(output_tas):
            inp = inputs[i]
            new_output_tas.append(tf_cond.cond(not_all_done, lambda: out_ta, lambda: out_ta.write(0, inp)))
        return (not_all_done, indices, inputs, new_output_tas)

    def _process_cond_stacked(self, conditions, indices, inputs, inputs_stacked, output_tas):
        num_outputs = len(self._outputs)
        not_all_done = math_ops.reduce_any(conditions)
        conditions_int = math_ops.cast(conditions, dtypes.int32)
        done_indices, new_indices = data_flow_ops.dynamic_partition(indices, conditions_int, 2)
        new_inputs = []
        new_output_tas = []
        for i, (inp, stacked) in enumerate(zip(inputs, inputs_stacked)):
            if stacked:
                done_inp, new_inp = data_flow_ops.dynamic_partition(inp, conditions_int, 2)
            else:
                done_inp = _stack(inp, [array_ops.size(done_indices)]).t
                new_inp = inp
            new_inputs.append(new_inp)
            if i < num_outputs:
                out_ta = output_tas[i]
                new_output_tas.append(out_ta.scatter(done_indices, done_inp))
        return (not_all_done, new_indices, new_inputs, new_output_tas)

    def _process_body(self, pfor_input, inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done):
        """Convert the body function."""

        def true_fn(control_inputs, body_pfor, body_output, stacked):
            """Converts the body function for all but last iteration.

      This essentially converts body_output. Additionally, it needs to handle
      any control dependencies on the NextIteration node. So it creates another
      Identity node with the converted dependencies.
      """
            converted_control_inp = []
            for x in control_inputs:
                for t in x.outputs:
                    converted_control_inp.append(body_pfor._convert_helper(t).t)
            if stacked:
                output = body_pfor.convert(body_output)
            else:
                output, convert_stacked, _ = body_pfor._convert_helper(body_output)
                assert convert_stacked == stacked, body_output
            with ops.control_dependencies(converted_control_inp):
                return array_ops.identity(output)
        body_pfor = self._init_pfor(pfor_input.pfor, new_indices, cond_stacked, new_inputs, inputs_stacked)
        new_outputs = []
        for i, (body_output, stacked) in enumerate(zip(self._body_outputs, inputs_stacked)):
            control_inp = self._next_iter_control_inputs[i]
            out_dtype = body_output.dtype
            new_output = tf_cond.cond(not_all_done, lambda: true_fn(control_inp, body_pfor, body_output, stacked), lambda: constant_op.constant([], dtype=out_dtype))
            new_outputs.append(new_output)
        return new_outputs

    def __call__(self, pfor_input):
        """Converter for the while_loop.

    The conversion of a while_loop is another while_loop.

    The arguments to this converted while_loop are as follows:
    not_all_done: Boolean scalar Tensor indicating if all the pfor iterations
      are done.
    indices: int32 1-D Tensor storing the id of the iterations that are not
      done.
    args: Remaining arguments. These can be divided into 3 categories:
      - First set of arguments are the tensors that correspond to the initial
        elements of self._enters. The elements that appear in original while
        loop's `loop_vars`.
      - The second set of arguments are the tensors that correspond to the
        remaining elements of self._enters. These are the tensors that directly
        enter the original while loop body.
       - Finally, the last set of arguments are TensorArrays. These TensorArrays
         correspond to the outputs of the original while_loop, i.e. to the
         elements in self._outputs. Each TensorArray has `PFor.loop_len`
         elements, i.e. the number of pfor iterations. At the end, the i'th
         element of each TensorArray will contain the output computed by the
         i'th iteration of pfor. Note that elements can be written into these
         tensors arrays in any order, depending on when the corresponding pfor
         iteration is done.
      If the original while_loop had `k` tensors in its `loop_vars` and its body
      directly captured `m` tensors, the `args` will contain `2 * k + m` values.

    In each iteration, the while_loop body recomputes the condition for all
    active pfor iterations to see which of them are now done. It then partitions
    all the inputs and passes them along to the converted body. Values for all
    the iterations that are done are written to TensorArrays indexed by the pfor
    iteration number. When all iterations are done, the TensorArrays are stacked
    to get the final value.

    Args:
      pfor_input: A PForInput object corresponding to the output of any Exit
        node from this while loop.

    Returns:
      List of converted outputs.
    """
        init_values, inputs_stacked, shape_invariants = self._create_init_values(pfor_input)
        cond_is_stacked = [None]

        def cond(not_all_done, *_):
            return not_all_done

        def body(not_all_done, indices, *args):
            num_enters = len(self._enters)
            inputs = args[:num_enters]
            output_tas = args[num_enters:]
            assert len(inputs) >= len(output_tas)
            assert len(inputs) == len(inputs_stacked)
            with ops.name_scope('while_cond'):
                cond_pfor = self._init_pfor(pfor_input.pfor, indices, cond_stacked=True, inputs=inputs, inputs_stacked=inputs_stacked)
                conditions, cond_stacked, _ = cond_pfor._convert_helper(self._condition)
                cond_is_stacked[0] = cond_stacked
            if not cond_stacked:
                not_all_done, new_indices, new_inputs, new_output_tas = self._process_cond_unstacked(conditions, indices, inputs, output_tas)
            else:
                not_all_done, new_indices, new_inputs, new_output_tas = self._process_cond_stacked(conditions, indices, inputs, inputs_stacked, output_tas)
            with ops.name_scope('while_body'):
                new_outputs = self._process_body(pfor_input, inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done)
            num_outputs = len(self._outputs)
            new_args = [not_all_done, new_indices] + new_outputs + list(new_inputs[num_outputs:]) + new_output_tas
            return tuple(new_args)
        while_outputs = while_loop.while_loop(cond, body, init_values, shape_invariants=shape_invariants)
        output_tas = while_outputs[-len(self._outputs):]
        outputs = []
        assert cond_is_stacked[0] is not None
        for inp_stacked, ta in zip(inputs_stacked, output_tas):
            if cond_is_stacked[0]:
                outputs.append(wrap(ta.stack(), True))
            else:
                outputs.append(wrap(ta.read(0), inp_stacked))
        return outputs