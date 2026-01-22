import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def _setup_constants(cls):
    for constant_info in cls.__info__.get_constants():
        name = constant_info.get_name()
        value = constant_info.get_value()
        setattr(cls, name, value)