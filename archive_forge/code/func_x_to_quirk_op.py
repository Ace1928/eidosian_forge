from typing import Any, Callable, cast, Dict, Optional, Union
import numpy as np
import sympy
from cirq import ops
def x_to_quirk_op(gate: ops.XPowGate) -> QuirkOp:
    return xyz_to_quirk_op('x', gate)