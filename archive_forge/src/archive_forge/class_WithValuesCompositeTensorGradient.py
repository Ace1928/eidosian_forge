import abc
import sys
from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import nest
class WithValuesCompositeTensorGradient(CompositeTensorGradient):
    """CompositeTensorGradient based on `T.values` and `T.with_values`."""

    def get_gradient_components(self, value):
        return value.values

    def replace_gradient_components(self, value, component_grads):
        return value.with_values(component_grads)