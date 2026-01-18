import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
A few representative samples of qubit maps.

    Only tests 10 combinations of Paulis to speed up testing.
    