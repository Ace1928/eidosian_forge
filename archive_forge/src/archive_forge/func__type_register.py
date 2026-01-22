import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
def _type_register(cls, namespace):
    if '__gtype__' in namespace:
        return
    if cls.__module__.startswith('gi.overrides.'):
        return
    _gi.type_register(cls, namespace.get('__gtype_name__'))