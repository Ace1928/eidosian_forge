from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def _resolve_signature(self):
    """Resolve signature.
        May have ambiguous case.
        """
    matches = []
    if self.scalarpos:
        for formaltys in self.typemap:
            match_map = []
            for i, (formal, actual) in enumerate(zip(formaltys, self.argtypes)):
                if actual is None:
                    actual = np.asarray(self.args[i]).dtype
                match_map.append(actual == formal)
            if all(match_map):
                matches.append(formaltys)
    if not matches:
        matches = []
        for formaltys in self.typemap:
            all_matches = all((actual is None or formal == actual for formal, actual in zip(formaltys, self.argtypes)))
            if all_matches:
                matches.append(formaltys)
    if not matches:
        raise TypeError("No matching version.  GPU ufunc requires array arguments to have the exact types.  This behaves like regular ufunc with casting='no'.")
    if len(matches) > 1:
        raise TypeError('Failed to resolve ufunc due to ambiguous signature. Too many untyped scalars. Use numpy dtype object to type tag.')
    self.argtypes = matches[0]