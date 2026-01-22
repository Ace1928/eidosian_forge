from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _make_reduce_func(self, reduce_func, input_dataset):
    """Make wrapping defun for reduce_func."""
    self._state_structure = self._init_func.output_structure
    state_types = self._init_func.output_types
    state_shapes = self._init_func.output_shapes
    state_classes = self._init_func.output_classes
    need_to_rerun = True
    while need_to_rerun:
        wrapped_func = structured_function.StructuredFunctionWrapper(reduce_func, self._transformation_name(), input_structure=(self._state_structure, input_dataset.element_spec), add_to_graph=False)
        for new_state_class, state_class in zip(nest.flatten(wrapped_func.output_classes), nest.flatten(state_classes)):
            if not issubclass(new_state_class, state_class):
                raise TypeError(f'Invalid `reducer`. The output class of the `reducer.reduce_func` {wrapped_func.output_classes}, does not match the class of the reduce state {self._state_classes}.')
        for new_state_type, state_type in zip(nest.flatten(wrapped_func.output_types), nest.flatten(state_types)):
            if new_state_type != state_type:
                raise TypeError(f'Invalid `reducer`. The element types for the new state {wrapped_func.output_types} do not match the element types of the old state {self._init_func.output_types}.')
        flat_state_shapes = nest.flatten(state_shapes)
        flat_new_state_shapes = nest.flatten(wrapped_func.output_shapes)
        weakened_state_shapes = [original.most_specific_compatible_shape(new) for original, new in zip(flat_state_shapes, flat_new_state_shapes)]
        need_to_rerun = False
        for original_shape, weakened_shape in zip(flat_state_shapes, weakened_state_shapes):
            if original_shape.ndims is not None and (weakened_shape.ndims is None or original_shape.as_list() != weakened_shape.as_list()):
                need_to_rerun = True
                break
        if need_to_rerun:
            state_shapes = nest.pack_sequence_as(self._init_func.output_shapes, weakened_state_shapes)
            self._state_structure = structure.convert_legacy_structure(state_types, state_shapes, state_classes)
    self._reduce_func = wrapped_func
    self._reduce_func.function.add_to_graph(ops.get_default_graph())