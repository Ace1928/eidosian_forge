import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
Create a new QASMQobj object from a dictionary.

        Args:
            data (dict): A dictionary representing the QasmQobj to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QasmQobj: The QasmQobj from the input dictionary.
        