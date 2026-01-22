import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
class StructMeta(type, MetaClassHelper):
    """Meta class used for GI Struct based types."""

    def __init__(cls, name, bases, dict_):
        super(StructMeta, cls).__init__(name, bases, dict_)
        g_type = cls.__info__.get_g_type()
        if g_type != TYPE_INVALID and g_type.pytype is not None:
            return
        cls._setup_fields()
        cls._setup_methods()
        for method_info in cls.__info__.get_methods():
            if method_info.is_constructor() and method_info.__name__ == 'new' and (not method_info.get_arguments() or cls.__info__.get_size() == 0):
                cls.__new__ = staticmethod(method_info)
                cls.__init__ = nothing
                break

    @property
    def __doc__(cls):
        if cls == StructMeta:
            return ''
        return generate_doc_string(cls.__info__)