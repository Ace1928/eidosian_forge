from collections import namedtuple
from collections.abc import Sequence
import numbers
import math
import re
import warnings
from traitlets import (
from ipywidgets import widget_serialization
from ipydatawidgets import DataUnion, NDArrayWidget, shape_constraints
import numpy as np
class Matrix3(Tuple):
    """A trait for a 9-tuple corresponding to a three.js Matrix3.
    """
    default_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    info_text = 'a three-by-three matrix (9 element tuple)'

    def __init__(self, trait=Undefined, default_value=Undefined, **kwargs):
        if trait is Undefined:
            trait = IEEEFloat()
        if default_value is Undefined:
            default_value = self.default_value
        else:
            self.default_value = default_value
        super(Matrix3, self).__init__(*(trait,) * 9, default_value=default_value, **kwargs)
        if isinstance(trait, IEEEFloat):
            self.metadata.setdefault('to_json', _ieee_tuple_to_json)