import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
def _cpmg_circuit(qubit: 'cirq.Qid', delay_var: sympy.Symbol, max_pulses: int) -> 'cirq.Circuit':
    """Creates a CPMG circuit for a given qubit.

    The circuit will look like:

      sqrt(Y) - wait(delay_var) - X - wait(2*delay_var) - ... - wait(delay_var)

    with max_pulses number of X gates.

    The X gates are paramterizd by 'pulse_N' symbols so that pulses can be
    turned on and off.  This is done to combine circuits with different pulses
    into the same paramterized circuit.
    """
    circuit = circuits.Circuit(ops.Y(qubit) ** 0.5, ops.wait(qubit, nanos=delay_var), ops.X(qubit))
    for n in range(max_pulses):
        pulse_n_on = sympy.Symbol(f'pulse_{n}')
        circuit.append(ops.wait(qubit, nanos=2 * delay_var * pulse_n_on))
        circuit.append(ops.X(qubit) ** pulse_n_on)
    circuit.append(ops.wait(qubit, nanos=delay_var))
    return circuit