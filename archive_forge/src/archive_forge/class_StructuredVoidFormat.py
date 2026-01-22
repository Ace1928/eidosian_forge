import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
class StructuredVoidFormat:
    """
    Formatter for structured np.void objects.

    This does not work on structured alias types like np.dtype(('i4', 'i2,i2')),
    as alias scalars lose their field information, and the implementation
    relies upon np.void.__getitem__.
    """

    def __init__(self, format_functions):
        self.format_functions = format_functions

    @classmethod
    def from_data(cls, data, **options):
        """
        This is a second way to initialize StructuredVoidFormat, using the raw data
        as input. Added to avoid changing the signature of __init__.
        """
        format_functions = []
        for field_name in data.dtype.names:
            format_function = _get_format_function(data[field_name], **options)
            if data.dtype[field_name].shape != ():
                format_function = SubArrayFormat(format_function, **options)
            format_functions.append(format_function)
        return cls(format_functions)

    def __call__(self, x):
        str_fields = [format_function(field) for field, format_function in zip(x, self.format_functions)]
        if len(str_fields) == 1:
            return '({},)'.format(str_fields[0])
        else:
            return '({})'.format(', '.join(str_fields))