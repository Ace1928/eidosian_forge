import warnings
from .api import _, is_validator, FancyValidator, Invalid, NoDefault
from . import declarative
from .exc import FERuntimeWarning
def _value_is_iterator(self, value):
    if isinstance(value, (bytes, str)):
        return False
    if isinstance(value, (list, tuple)):
        return True
    try:
        for _v in value:
            break
        return True
    except TypeError:
        return False