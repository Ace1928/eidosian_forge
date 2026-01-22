from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
class FormalArgument(QuilAtom):
    """
    Representation of a formal argument associated with a DEFCIRCUIT or DEFGATE ... AS PAULI-SUM
    or DEFCAL form.
    """

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError('Formal arguments must be named by a string.')
        self.name = name

    def out(self) -> str:
        return str(self)

    @property
    def index(self) -> NoReturn:
        raise RuntimeError(f'Cannot derive an index from a FormalArgument {self}')

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<FormalArgument {self.name}>'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FormalArgument) and other.name == self.name