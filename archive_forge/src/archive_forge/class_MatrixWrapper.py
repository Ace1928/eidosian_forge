import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
class MatrixWrapper:

    def __init__(self, name, attribute):
        self.name = name
        self.attribute = attribute
        self._attribute = '_' + attribute

    def __get__(self, obj, objtype):
        matrix = getattr(obj, self._attribute, None)
        return matrix

    def __set__(self, obj, value):
        value = np.asarray(value, order='F')
        shape = obj.shapes[self.attribute]
        if len(shape) == 3:
            value = self._set_matrix(obj, value, shape)
        else:
            value = self._set_vector(obj, value, shape)
        setattr(obj, self._attribute, value)
        obj.shapes[self.attribute] = value.shape

    def _set_matrix(self, obj, value, shape):
        if value.ndim == 1 and shape[0] == 1 and (value.shape[0] == shape[1]):
            value = value[None, :]
        validate_matrix_shape(self.name, value.shape, shape[0], shape[1], obj.nobs)
        if value.ndim == 2:
            value = np.array(value[:, :, None], order='F')
        return value

    def _set_vector(self, obj, value, shape):
        validate_vector_shape(self.name, value.shape, shape[0], obj.nobs)
        if value.ndim == 1:
            value = np.array(value[:, None], order='F')
        return value