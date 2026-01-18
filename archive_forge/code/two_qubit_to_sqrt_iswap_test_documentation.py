import itertools
import numpy as np
import pytest
import cirq
import sympy
Returns several unitaries in the neighborhood of u to test for numerical
    corner cases near critical values.