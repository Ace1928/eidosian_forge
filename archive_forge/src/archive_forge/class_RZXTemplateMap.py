from enum import Enum
from typing import List, Dict
from qiskit.circuit.library.templates import rzx
class RZXTemplateMap(Enum):
    """Mapping of instruction name to decomposition template."""
    ZZ1 = rzx.rzx_zz1()
    ZZ2 = rzx.rzx_zz2()
    ZZ3 = rzx.rzx_zz3()
    YZ = rzx.rzx_yz()
    XZ = rzx.rzx_xz()
    CY = rzx.rzx_cy()