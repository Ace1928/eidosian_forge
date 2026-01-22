from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
@validate('array')
def _validate_array(self, proposal):
    """Validate array against external validators (instance only)

        This allows others to add constraints on the array of this
        widget dynamically. Internal use is so that a constrained
        DataUnion can validate the array of a widget set to itself.
        """
    value = proposal['value']
    for validator in self._instance_validators:
        value = validator(value)
    return value