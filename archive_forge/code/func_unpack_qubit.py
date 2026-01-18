from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def unpack_qubit(qubit: Union[QubitDesignator, FormalArgument]) -> Union[Qubit, QubitPlaceholder, FormalArgument]:
    """
    Get a qubit from an object.

    :param qubit: the qubit designator to unpack.
    :return: A Qubit or QubitPlaceholder instance
    """
    if isinstance(qubit, int):
        return Qubit(qubit)
    elif isinstance(qubit, Qubit):
        return qubit
    elif isinstance(qubit, QubitPlaceholder):
        return qubit
    elif isinstance(qubit, FormalArgument):
        return qubit
    else:
        raise TypeError('qubit should be an int or Qubit or QubitPlaceholder instance')