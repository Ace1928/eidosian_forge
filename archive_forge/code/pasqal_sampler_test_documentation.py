from unittest.mock import patch
import copy
import numpy as np
import sympy
import pytest
import cirq
import cirq_pasqal
Test running a sweep.

    Encodes a random binary number in the qubits, sweeps between odd and even
    without noise and checks if the results match.
    