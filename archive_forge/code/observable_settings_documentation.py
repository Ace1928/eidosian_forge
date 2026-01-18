import dataclasses
import numbers
from typing import (
import sympy
from cirq import ops, value, protocols
An encapsulation of all the specifications for one run of a
    quantum processor.

    This includes the maximal input-output setting (which may result in many
    observables being measured if they are consistent with `max_setting`) and
    a set of circuit parameters if the circuit is parameterized.
    