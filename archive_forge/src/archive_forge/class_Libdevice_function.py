from numba.cuda import libdevice, libdevicefuncs
from numba.core.typing.templates import ConcreteTemplate, Registry
class Libdevice_function(ConcreteTemplate):
    cases = [libdevicefuncs.create_signature(retty, args)]