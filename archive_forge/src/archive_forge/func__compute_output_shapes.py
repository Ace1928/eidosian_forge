import logging
from .base_module import BaseModule
from ..initializer import Uniform
from .. import ndarray as nd
def _compute_output_shapes(self):
    """Computes the shapes of outputs. As a loss module with outputs, we simply
        output whatever we receive as inputs (i.e. the scores).
        """
    return [(self._name + '_output', self._data_shapes[0][1])]