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
class WhileV2:
    """Object for vectorizing V2 while_loop op."""

    def __init__(self, pfor_input):
        self._pfor_input = pfor_input
        self._pfor = pfor_input.pfor
        cond_func_name = pfor_input.get_attr('cond').name
        self._cond_func = pfor_input.op.graph._get_function(compat.as_bytes(cond_func_name))
        body_func_name = pfor_input.get_attr('body').name
        self._body_func = pfor_input.op.graph._get_function(compat.as_bytes(body_func_name))
        if self._cond_func is None or self._body_func is None:
            raise ValueError(f'Error extracting cond and body functions for op {self._pfor_input.op}.')
        self._body_pass_through_indices = set()
        for i, (inp, out) in enumerate(zip(self._body_func.graph.inputs, self._body_func.graph.outputs)):
            if id(inp) == id(out):
                self._body_pass_through_indices.add(i)
        self._parallel_iterations = self._pfor_input.get_attr('parallel_iterations')

    def _output_shapes(self):
        output_shapes = [out.shape for out in self._pfor_input.op.outputs]
        shapes = self._pfor_input.get_attr('output_shapes')
        if not shapes:
            shapes = [tensor_shape.TensorShape(None) for _ in output_shapes]
        else:
            shapes = [tensor_shape.TensorShape(shape) for shape in shapes]
        for i, shape in enumerate(shapes):
            shape = shape.merge_with(output_shapes[i])
            pfor_input = self._pfor_input.input(i)
            if pfor_input.is_stacked:
                if _is_variant_with_internal_stacking(pfor_input.t):
                    shape = tensor_shape.TensorShape([]).concatenate(shape)
                else:
                    shape = tensor_shape.TensorShape([None]).concatenate(shape)
            output_shapes[i] = shape
        assert len(output_shapes) == self._pfor_input.num_inputs
        return output_shapes

    def _init_values(self):
        """Create arguments passed to converted while_loop."""
        loop_len = self._pfor.loop_len_vector[0]
        inputs = []
        output_tas = []
        with ops.name_scope('while_init'):
            for inp in self._pfor_input.inputs:
                inputs.append(inp.t)
                variant_type_id = _variant_type_id(inp.t)
                if variant_type_id in _INTERNAL_STACKING_TYPE_IDS:
                    if variant_type_id != full_type_pb2.TFT_ARRAY:
                        raise NotImplementedError(f'While loop conversion is only supported for TensorLists. Got another variant {inp.t}, probably an optional. Please file a bug.')
                    element_shape = list_ops.tensor_list_element_shape(inp.t, dtypes.int32)
                    if inp.is_stacked:
                        element_shape = tf_cond.cond(math_ops.equal(array_ops.rank(element_shape), 0), lambda: element_shape, lambda: element_shape[1:])
                    dtype = _parse_variant_shapes_and_types(inp.t)[0].dtype

                    def _init_loop_body(index, output_ta):
                        output_ta = output_ta.write(index, list_ops.tensor_list_reserve(element_shape, loop_len, dtype))
                        return (index + 1, output_ta)
                    length = list_ops.tensor_list_length(inp.t)
                    output_ta = tensor_array_ops.TensorArray(inp.t.dtype, size=length, dynamic_size=True, infer_shape=False)
                    _, output_ta = while_loop.while_loop(lambda index, _: index < length, _init_loop_body, [0, output_ta])
                else:
                    output_ta = tensor_array_ops.TensorArray(inp.t.dtype, size=loop_len, dynamic_size=False, infer_shape=True)
                output_tas.append(output_ta)
        indices = math_ops.range(self._pfor.loop_len_vector[0]) if self._pfor.all_indices_partitioned else self._pfor.all_indices
        return [True, indices] + inputs + output_tas

    def _process_cond_unstacked(self, conditions, indices, inputs, output_tas):
        """Handles case when condition is pfor loop invariant."""
        not_all_done = array_ops.reshape(conditions, [])
        return (not_all_done, indices, inputs, output_tas)

    def _process_cond_stacked(self, conditions, indices, inputs, inputs_stacked, output_tas):
        """Handles case when condition is pfor loop dependent."""
        not_all_done = math_ops.reduce_any(conditions)
        conditions_int = math_ops.cast(conditions, dtypes.int32)
        done_indices, new_indices = data_flow_ops.dynamic_partition(indices, conditions_int, 2)
        new_inputs = []
        new_output_tas = []
        for i, (inp, stacked) in enumerate(zip(inputs, inputs_stacked)):
            pass_through = i in self._body_pass_through_indices
            if not pass_through and _variant_type_id(inp) == full_type_pb2.TFT_ARRAY:
                shape_and_type = _parse_variant_shapes_and_types(inp)[0]
                element_shape = list_ops.tensor_list_element_shape(inp, dtypes.int32)
                user_list_len = list_ops.tensor_list_length(inp)

                def _split_vectorized_ta_element(index, new_inp, new_out_ta):
                    elem = list_ops.tensor_list_get_item(inp, index, shape_and_type.dtype, element_shape)
                    if stacked:
                        done_elem, new_elem = data_flow_ops.dynamic_partition(elem, conditions_int, 2)
                        new_inp = list_ops.tensor_list_set_item(new_inp, index, new_elem)
                    else:
                        done_elem = _stack(elem, [array_ops.size(done_indices)]).t
                    done_accum = new_out_ta.read(index)
                    done_accum = list_ops.tensor_list_scatter(tensor=done_elem, indices=done_indices, input_handle=done_accum)
                    new_out_ta = new_out_ta.write(index, done_accum)
                    return (index + 1, new_inp, new_out_ta)
                length = list_ops.tensor_list_length(inp)
                new_inp = list_ops.tensor_list_reserve(tensor_shape.TensorShape([None]) + tensor_shape.TensorShape(shape_and_type.shape)[1:], user_list_len, shape_and_type.dtype)
                _, new_inp, out_ta = while_loop.while_loop(lambda index, unused_new_inp, unused_new_out_ta: index < length, _split_vectorized_ta_element, [0, new_inp, output_tas[i]])
            else:
                if stacked:
                    done_inp, new_inp = data_flow_ops.dynamic_partition(inp, conditions_int, 2)
                else:
                    if not pass_through:
                        done_inp = _stack(inp, [array_ops.size(done_indices)]).t
                    new_inp = inp
                out_ta = output_tas[i]
                if not pass_through:
                    out_ta = out_ta.scatter(done_indices, done_inp)
            new_inputs.append(new_inp)
            new_output_tas.append(out_ta)
        assert len(new_output_tas) == len(output_tas)
        assert len(new_inputs) == len(inputs)
        return (not_all_done, new_indices, new_inputs, new_output_tas)

    def _process_body(self, inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done):
        """Convert the body function."""
        mismatching_stacked_indices = []

        def true_fn():
            """Converts the body function for all but last iteration."""
            wrapped_inputs = [wrap(inp, stacked) for inp, stacked in zip(new_inputs, inputs_stacked)]
            while True:
                if self._pfor.all_indices_partitioned:
                    indices = array_ops.gather(self._pfor.all_indices, new_indices)
                else:
                    indices = new_indices
                body_pfor = PFor(loop_var=self._pfor.loop_var, loop_len=array_ops.size(new_indices), pfor_ops=self._body_func.graph.get_operations(), fallback_to_while_loop=self._pfor.fallback_to_while_loop, all_indices=indices, all_indices_partitioned=self._pfor.all_indices_partitioned or cond_stacked, pfor_config=self._pfor.pfor_config)
                stacking_mismatch = False
                outputs = _convert_function_call(self._body_func, body_pfor, wrapped_inputs)
                for i, (out, inp) in enumerate(zip(outputs, wrapped_inputs)):
                    if out.is_stacked != inp.is_stacked:
                        stacking_mismatch = True
                        mismatching_stacked_indices.append(i)
                        stacked = _stack(inp.t, [array_ops.size(new_indices)])
                        if inp.t.dtype == dtypes.variant:
                            stacked = wrap(_tile_variant_with_length(stacked.t, [array_ops.size(new_indices)]))
                        wrapped_inputs[i] = stacked
                if not stacking_mismatch:
                    if mismatching_stacked_indices:
                        with ops.control_dependencies([control_flow_assert.Assert(False, ['pfor ERROR: this branch should never execute'])]):
                            return [array_ops.identity(x) for x in new_inputs]
                    else:
                        return [out.t for out in outputs]
        return (tf_cond.cond(not_all_done, true_fn, lambda: list(new_inputs)), mismatching_stacked_indices)

    def __call__(self):
        """Converter for the V2 while_loop.

    The conversion of a while_loop is another while_loop.

    The arguments to this converted while_loop are as follows:
    not_all_done: Boolean scalar Tensor indicating if all the pfor iterations
      are done.
    indices: int32 1-D Tensor storing the id of the pfor iterations that are not
      done.
    args: Remaining arguments. These can be divided into 2 categories:
      - The first set of arguments correspond one-to-one to the inputs to the
        unvectorized while_loop.
      - The second set are TensorArrays, corresponding one-to-one to each output
        of the unvectorized while_loop. Each TensorArray has `PFor.loop_len`
        elements, i.e. the number of pfor iterations. At the end, the i'th
        element of each TensorArray will contain the output computed by the i'th
        iteration of pfor. Note that elements can be written into these tensors
        arrays in any order, depending on when the corresponding pfor iteration
        is done.
    In each iteration, the while_loop body recomputes the condition for all
    active pfor iterations to see which of them are now done. It then partitions
    all the inputs and passes them along to the converted body. Values for all
    the iterations that are done are written to TensorArrays indexed by the pfor
    iteration number. When all iterations are done, the TensorArrays are stacked
    to get the final value.

    Returns:
      List of converted outputs.
    """
        output_shapes = self._output_shapes()
        cond_is_stacked = [None]
        indices_to_stack = []

        def cond(not_all_done, *_):
            return not_all_done

        def body(not_all_done, indices, *args):
            num_inputs = self._pfor_input.num_inputs
            inputs = args[:num_inputs]
            output_tas = args[num_inputs:]
            inputs_stacked = [x.is_stacked for x in self._pfor_input.inputs]
            assert len(inputs) >= len(output_tas)
            assert len(inputs) == len(inputs_stacked)
            with ops.name_scope('while_cond'):
                cond_pfor = PFor(loop_var=self._pfor.loop_var, loop_len=array_ops.size(indices), pfor_ops=self._cond_func.graph.get_operations(), fallback_to_while_loop=self._pfor.fallback_to_while_loop, all_indices=indices, all_indices_partitioned=True, pfor_config=self._pfor.pfor_config)
                wrapped_inputs = [wrap(inp, stacked) for inp, stacked in zip(inputs, inputs_stacked)]
                conditions, cond_stacked, _ = _convert_function_call(self._cond_func, cond_pfor, wrapped_inputs)[0]
                cond_is_stacked[0] = cond_stacked
            if not cond_stacked:
                not_all_done, new_indices, new_inputs, new_output_tas = self._process_cond_unstacked(conditions, indices, inputs, output_tas)
            else:
                not_all_done, new_indices, new_inputs, new_output_tas = self._process_cond_stacked(conditions, indices, inputs, inputs_stacked, output_tas)
            with ops.name_scope('while_body'):
                new_outputs, mismatching_stacked_indices = self._process_body(inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done)
            indices_to_stack[:] = mismatching_stacked_indices
            for i, new_output in enumerate(new_outputs):
                new_output.set_shape(output_shapes[i])
            new_args = [not_all_done, new_indices] + new_outputs + list(new_output_tas)
            return tuple(new_args)

        @def_function.function
        def while_fn():
            init_values = self._init_values()
            ta_shape_invariants = [tensor_shape.TensorShape([]) for _ in self._pfor_input.outputs]
            shape_invariants = [tensor_shape.TensorShape([]), tensor_shape.TensorShape([None])] + output_shapes + ta_shape_invariants
            while_outputs = while_loop.while_loop(cond, body, init_values, shape_invariants=shape_invariants, parallel_iterations=self._parallel_iterations)
            if indices_to_stack:
                return while_outputs
            else:
                num_inputs = self._pfor_input.num_inputs
                new_inputs = while_outputs[2:num_inputs + 2]
                output_tas = while_outputs[num_inputs + 2:]
                assert cond_is_stacked[0] is not None
                outputs = []
                for i, inp in enumerate(new_inputs):
                    if cond_is_stacked[0]:
                        if i in self._body_pass_through_indices:
                            outputs.append(init_values[i + 2])
                        else:
                            ta = output_tas[i]
                            if _variant_type_id(inp) == full_type_pb2.TFT_ARRAY:
                                shape_and_type = _parse_variant_shapes_and_types(inp)[0]
                                length = list_ops.tensor_list_length(inp)

                                def _stack_loop_body(index, output_list):
                                    current_value = ta.read(index)
                                    output_list = list_ops.tensor_list_set_item(output_list, index, list_ops.tensor_list_stack(current_value, shape_and_type.dtype))
                                    return (index + 1, output_list)
                                output_list = list_ops.tensor_list_reserve(tensor_shape.TensorShape(shape_and_type.shape), length, shape_and_type.dtype)
                                _, output_list = while_loop.while_loop(lambda index, _: index < length, _stack_loop_body, [0, output_list])
                                outputs.append(output_list)
                            else:
                                outputs.append(ta.stack())
                    else:
                        outputs.append(inp)
                return outputs
        _ = while_fn.get_concrete_function()
        if indices_to_stack:
            self._pfor_input.stack_inputs(stack_indices=indices_to_stack, tile_variants=True)
            return self()
        else:
            outputs = while_fn()
            wrapped_outputs = []
            for i, (out, inp) in enumerate(zip(outputs, self._pfor_input.inputs)):
                if i not in self._body_pass_through_indices and cond_is_stacked[0]:
                    wrapped_outputs.append(wrap(out, True))
                else:
                    wrapped_outputs.append(wrap(out, inp.is_stacked))
            return wrapped_outputs