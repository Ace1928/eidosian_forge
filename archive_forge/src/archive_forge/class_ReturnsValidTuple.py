import pytest
import numpy as np
import cirq
class ReturnsValidTuple(cirq.SupportsMixture):

    def _mixture_(self):
        return ((0.4, 'a'), (0.6, 'b'))

    def _has_mixture_(self):
        return True