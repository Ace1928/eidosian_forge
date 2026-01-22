import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class DeprecatedInterfaceMeta(InterfaceMeta):

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        classdict.setdefault('_plugins', _deprecated_plugin_dict(name, classdict))
        return super().__new__(cls, name, bases, classdict, *args, **kwargs)