import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
class GateProperties:
    """Class representing a gate's properties

    Attributes:
        qubits: qubits.
        gate: gate.
        parameters: parameters.
    """
    _data = {}

    def __init__(self, qubits, gate, parameters, **kwargs):
        """Initialize a new :class:`GateProperties` object

        Args:
            qubits (list): A list of integers representing qubits
            gate (str): The gates name
            parameters (list): List of :class:`Nduv` objects for the
                name-date-unit-value for the gate
            kwargs: Optional additional fields
        """
        self._data = {}
        self.qubits = qubits
        self.gate = gate
        self.parameters = parameters
        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f'Attribute {name} is not defined') from ex

    @classmethod
    def from_dict(cls, data):
        """Create a new Gate object from a dictionary.

        Args:
            data (dict): A dictionary representing the Gate to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            GateProperties: The Nduv from the input dictionary.
        """
        in_data = {}
        for key, value in data.items():
            if key == 'parameters':
                in_data[key] = list(map(Nduv.from_dict, value))
            else:
                in_data[key] = value
        return cls(**in_data)

    def to_dict(self):
        """Return a dictionary format representation of the BackendStatus.

        Returns:
            dict: The dictionary form of the Gate.
        """
        out_dict = {}
        out_dict['qubits'] = self.qubits
        out_dict['gate'] = self.gate
        out_dict['parameters'] = [x.to_dict() for x in self.parameters]
        out_dict.update(self._data)
        return out_dict

    def __eq__(self, other):
        if isinstance(other, GateProperties):
            if self.to_dict() == other.to_dict():
                return True
        return False