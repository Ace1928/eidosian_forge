import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class DeprecatedInterface(Interface, metaclass=DeprecatedInterfaceMeta):
    pass