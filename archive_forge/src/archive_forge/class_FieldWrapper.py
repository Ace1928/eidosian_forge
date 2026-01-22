import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class FieldWrapper:
    """Object to wrap field to pass to `multiprocessing.Pool`."""

    def __init__(self, field, field_args):
        self.field = field
        self.field_args = field_args

    def func(self, v_x_a):
        try:
            v_f = self.field(v_x_a, *self.field_args)
        except Exception:
            v_f = np.inf
        if np.isnan(v_f):
            v_f = np.inf
        return v_f