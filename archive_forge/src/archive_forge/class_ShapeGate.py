import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class ShapeGate(cirq.Gate):

    def _qid_shape_(self):
        return (1, 2, 3, 4)