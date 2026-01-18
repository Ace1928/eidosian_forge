from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
Returns a controlled `SWAP` with one additional control.

        The `controlled` method of the `Gate` class, of which this class is a
        child, returns a `ControlledGate` with `sub_gate = self`. This method
        overrides this behavior to return a `ControlledGate` with
        `sub_gate = SWAP`.
        