import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class LMul:

    def __mul__(self, other):
        return 'Yay!'