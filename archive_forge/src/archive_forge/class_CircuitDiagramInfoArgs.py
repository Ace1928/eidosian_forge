import re
from fractions import Fraction
from typing import (
import numpy as np
import sympy
from typing_extensions import Protocol
from cirq import protocols, value
from cirq._doc import doc_private
@value.value_equality
class CircuitDiagramInfoArgs:
    """A request for information on drawing an operation in a circuit diagram.

    Attributes:
        known_qubits: The qubits the gate is being applied to. None means this
            information is not known by the caller.
        known_qubit_count: The number of qubits the gate is being applied to
            None means this information is not known by the caller.
        use_unicode_characters: If true, the wire symbols are permitted to
            include unicode characters (as long as they work well in fixed
            width fonts). If false, use only ascii characters. ASCII is
            preferred in cases where UTF8 support is done poorly, or where
            the fixed-width font being used to show the diagrams does not
            properly handle unicode characters.
        precision: The number of digits after the decimal to show for numbers in
            the text diagram. None means use full precision.
        label_map: The map from label entities to diagram positions.
        include_tags: Whether to print tags from TaggedOperations.
        transpose: Whether the circuit is to be drawn with time from left to
            right (transpose is False), or from top to bottom.
    """
    UNINFORMED_DEFAULT: 'CircuitDiagramInfoArgs'

    def __init__(self, known_qubits: Optional[Iterable['cirq.Qid']], known_qubit_count: Optional[int], use_unicode_characters: bool, precision: Optional[int], label_map: Optional[Dict['cirq.LabelEntity', int]], include_tags: bool=True, transpose: bool=False) -> None:
        self.known_qubits = None if known_qubits is None else tuple(known_qubits)
        self.known_qubit_count = known_qubit_count
        self.use_unicode_characters = use_unicode_characters
        self.precision = precision
        self.label_map = label_map
        self.include_tags = include_tags
        self.transpose = transpose

    def _value_equality_values_(self) -> Any:
        return (self.known_qubits, self.known_qubit_count, self.use_unicode_characters, self.precision, None if self.label_map is None else tuple(sorted(self.label_map.items(), key=lambda e: e[0])), self.include_tags, self.transpose)

    def __repr__(self) -> str:
        return f'cirq.CircuitDiagramInfoArgs(known_qubits={self.known_qubits!r}, known_qubit_count={self.known_qubit_count!r}, use_unicode_characters={self.use_unicode_characters!r}, precision={self.precision!r}, label_map={self.label_map!r}, include_tags={self.include_tags!r}, transpose={self.transpose!r})'

    def format_real(self, val: Union[sympy.Basic, int, float]) -> str:
        if isinstance(val, sympy.Basic):
            return str(val)
        if val == int(val):
            return str(int(val))
        if self.precision is None:
            return str(val)
        return f'{float(val):.{self.precision}}'

    def format_complex(self, val: Union[sympy.Basic, int, float, 'cirq.TParamValComplex']) -> str:
        if isinstance(val, sympy.Basic):
            return str(val)
        c = complex(val)
        joiner = '+'
        abs_imag = c.imag
        if abs_imag < 0:
            joiner = '-'
            abs_imag *= -1
        imag_str = '' if abs_imag == 1 else self.format_real(abs_imag)
        return f'{self.format_real(c.real)}{joiner}{imag_str}i'

    def format_radians(self, radians: Union[sympy.Basic, int, float]) -> str:
        """Returns angle in radians as a human-readable string."""
        if protocols.is_parameterized(radians):
            return str(radians)
        unit = 'π' if self.use_unicode_characters else 'pi'
        if radians == np.pi:
            return unit
        if radians == 0:
            return '0'
        if radians == -np.pi:
            return f'-{unit}'
        if self.precision is not None and (not isinstance(radians, sympy.Basic)):
            quantity = self.format_real(radians / np.pi)
            return quantity + unit
        return repr(radians)

    def copy(self):
        return self.__class__(known_qubits=self.known_qubits, known_qubit_count=self.known_qubit_count, use_unicode_characters=self.use_unicode_characters, precision=self.precision, label_map=self.label_map, transpose=self.transpose)

    def with_args(self, **kwargs):
        args = self.copy()
        for arg_name, val in kwargs.items():
            setattr(args, arg_name, val)
        return args