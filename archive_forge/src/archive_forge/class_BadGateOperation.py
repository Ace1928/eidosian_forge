from typing import Optional, Sequence, Union, Collection, Tuple, List
import pytest
import numpy as np
import cirq
from cirq.ops import control_values as cv
class BadGateOperation(cirq.GateOperation):

    def controlled_by(self, *control_qubits: 'cirq.Qid', control_values: Optional[Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None) -> 'cirq.Operation':
        return cirq.ControlledOperation(control_qubits, self, control_values)