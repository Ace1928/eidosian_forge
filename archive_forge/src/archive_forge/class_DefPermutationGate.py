import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DefPermutationGate(DefGate):

    def __init__(self, name: str, permutation: Union[List[Union[int, np.int_]], np.ndarray]):
        if not isinstance(name, str):
            raise TypeError('Gate name must be a string')
        if name in RESERVED_WORDS:
            raise ValueError(f"Cannot use {name} for a gate name since it's a reserved word")
        if not isinstance(permutation, (list, np.ndarray)):
            raise ValueError(f'Permutation must be a list or NumPy array, got value of type {type(permutation)}')
        permutation = np.asarray(permutation)
        ndim = permutation.ndim
        if 1 != ndim:
            raise ValueError(f'Permutation must have dimension 1, got {permutation.ndim}')
        elts = permutation.shape[0]
        if 0 != elts & elts - 1:
            raise ValueError(f'Dimension of permutation must be a power of 2, got {elts}')
        self.name = name
        self.permutation = permutation
        self.parameters = None

    def out(self) -> str:
        body = ', '.join([str(p) for p in self.permutation])
        return f'DEFGATE {self.name} AS PERMUTATION:\n    {body}'

    def num_args(self) -> int:
        """
        :return: The number of qubit arguments the gate takes.
        """
        return int(np.log2(len(self.permutation)))