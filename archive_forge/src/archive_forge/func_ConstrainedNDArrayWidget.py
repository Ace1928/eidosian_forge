from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
def ConstrainedNDArrayWidget(*validators, **kwargs):
    import warnings
    warnings.warn('ConstrainedNDArrayWidget is deprecated, use create_constrained_arraywidget instead')
    return create_constrained_arraywidget(*validators, **kwargs)