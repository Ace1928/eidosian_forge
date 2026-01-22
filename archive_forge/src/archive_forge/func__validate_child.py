from functools import partial
import numpy as np
from traitlets import Union, Instance, Undefined, TraitError
from .serializers import data_union_serialization
from .traits import NDArray
from .widgets import NDArrayWidget, NDArrayBase, NDArraySource
def _validate_child(self, obj, value):
    try:
        return self.validate(obj, value)
    except TraitError:
        raise TraitError('Widget data is constrained by its use in %r.' % obj)