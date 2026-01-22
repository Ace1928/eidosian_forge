from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
class AspenQubit(OctagonalQubit):

    def __init__(self, octagon: int, octagon_position: int):
        super(AspenQubit, self).__init__(octagon_position)
        self._octagon = octagon
        self.index = octagon * 10 + octagon_position

    @property
    def octagon(self):
        return self._octagon

    def _comparison_key(self):
        return (self.octagon, self.index)

    @property
    def x(self) -> float:
        """Returns the horizontal position of the qubit, assuming each side of
        the octagon has length 1.

        Returns:
            The horizontal position of the qubit.

        Raises:
            ValueError: Octagon position is invalid.
        """
        octagon_left_most_position = self.octagon * (2 + sqrt(2))
        if self.octagon_position in {5, 6}:
            return octagon_left_most_position
        if self.octagon_position in {4, 7}:
            return octagon_left_most_position + 1 / sqrt(2)
        if self.octagon_position in {0, 3}:
            return octagon_left_most_position + 1 + 1 / sqrt(2)
        if self.octagon_position in {1, 2}:
            return octagon_left_most_position + 1 + sqrt(2)
        raise ValueError(f'invalid octagon position {self.octagon_position}')

    def distance(self, other: cirq.Qid) -> float:
        """Returns the distance between two qubits.

        Args:
            other: An AspenQubit to which we are measuring distance.

        Returns:
            The distance between two qubits.
        Raises:
            TypeError: other qubit must be AspenQubit.
        """
        if type(other) != AspenQubit:
            raise TypeError('can only measure distance from other Aspen qubits')
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def to_grid_qubit(self) -> cirq.GridQubit:
        """Converts `AspenQubit` to `cirq.GridQubit`.

        Returns:
            The equivalent GridQubit.

        Raises:
            ValueError: AspenQubit cannot be converted to GridQubit.
        """
        for grid_qubit, aspen_index in _grid_qubit_mapping.items():
            if self.index == aspen_index:
                return grid_qubit
        raise ValueError(f'cannot use {self} as a GridQubit')

    def to_named_qubit(self) -> cirq.NamedQubit:
        """Converts `AspenQubit` to `cirq.NamedQubit`.

        Returns:
            The equivalent NamedQubit.
        """
        return cirq.NamedQubit(str(self.index))

    @staticmethod
    def from_grid_qubit(grid_qubit: cirq.GridQubit) -> 'AspenQubit':
        """Converts `cirq.GridQubit` to `AspenQubit`.

        Returns:
            The equivalent AspenQubit.

        Raises:
            ValueError: GridQubit cannot be converted to AspenQubit.
        """
        if grid_qubit in _grid_qubit_mapping:
            return AspenQubit.from_aspen_index(_grid_qubit_mapping[grid_qubit])
        raise ValueError(f'{grid_qubit} is not convertible to Aspen qubit')

    @staticmethod
    def from_named_qubit(qubit: cirq.NamedQubit) -> 'AspenQubit':
        """Converts `cirq.NamedQubit` to `AspenQubit`.

        Returns:
            The equivalent AspenQubit.

        Raises:
            ValueError: NamedQubit cannot be converted to AspenQubit.
            UnsupportedQubit: If the supplied qubit is not a named qubit with an octagonal
                index.
        """
        try:
            index = int(qubit.name)
            return AspenQubit.from_aspen_index(index)
        except ValueError:
            raise UnsupportedQubit('Aspen devices only support named qubits by octagonal index')

    @staticmethod
    def from_aspen_index(index: int) -> 'AspenQubit':
        """Initializes an `AspenQubit` at the given index. See `OctagonalQubit` to understand
        OctagonalQubit indexing.

        Args:
            index: The index at which to initialize the `AspenQubit`.

        Returns:
            The AspenQubit with requested index.

        Raises:
            ValueError: index is not a valid octagon position.
        """
        octagon_position = index % 10
        octagon = np.floor(index / 10.0)
        return AspenQubit(octagon, octagon_position)

    def __repr__(self):
        return f'cirq_rigetti.AspenQubit(octagon={self.octagon}, octagon_position={self.octagon_position})'

    def __str__(self):
        return f'({self.octagon}, {self.octagon_position})'

    def _json_dict_(self):
        return {'octagon': self.octagon, 'octagon_position': self.octagon_position}