import unittest.mock as mock
import numpy as np
import sympy
import cirq
class XContainer(cirq.Gate):

    def _decompose_(self, qs):
        return [cirq.X(*qs)]

    def _qid_shape_(self):
        pass