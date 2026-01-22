from typing import AbstractSet, Any, Optional, Tuple
import numpy as np
import cirq
from cirq._compat import proper_repr
def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
    s = f'/‾‾({self.hold_time}@{self.coupling_mhz}MHz)‾‾\\'
    return (s, s)