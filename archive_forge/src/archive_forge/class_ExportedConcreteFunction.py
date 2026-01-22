import gc
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.trackable import base as trackable
class ExportedConcreteFunction(trackable.Trackable):
    """A callable class that uses captures from the exported SavedModel graph."""
    __slots__ = ('function', 'tensor_map')

    def __init__(self, function, tensor_map):
        self.function = function
        self.tensor_map = tensor_map

    def __call__(self, *args, **kwargs):
        bound_arguments = function_type_utils.canonicalize_function_inputs(args, kwargs, self.function._function_type)
        filtered_flat_args = self.function._function_type.unpack_inputs(bound_arguments)
        export_captures = _map_captures_to_created_tensors(self.function.graph.captures, self.tensor_map, self.function)
        return self.function._call_flat(filtered_flat_args, export_captures)