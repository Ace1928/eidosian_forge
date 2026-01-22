import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
def _strip_modifiers(gate: Gate, limit: Optional[int]=None) -> Gate:
    """
    Remove modifiers from :py:class:`Gate`.

    This function removes up to ``limit`` gate modifiers from the given gate,
    starting from the leftmost gate modifier.

    :param gate: A gate.
    :param limit: An upper bound on how many modifiers to remove.
    """
    if limit is None:
        limit = len(gate.modifiers)
    qubit_index = 0
    param_index = len(gate.params)
    for m in gate.modifiers[:limit]:
        if m == 'CONTROLLED':
            qubit_index += 1
        elif m == 'FORKED':
            if param_index % 2 != 0:
                raise ValueError('FORKED gate has an invalid number of parameters.')
            param_index //= 2
            qubit_index += 1
        elif m == 'DAGGER':
            pass
        else:
            raise TypeError('Unsupported gate modifier {}'.format(m))
    stripped = Gate(gate.name, gate.params[:param_index], gate.qubits[qubit_index:])
    stripped.modifiers = gate.modifiers[limit:]
    return stripped