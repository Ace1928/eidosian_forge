from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
def test_with_params():
    assert gateset.with_params() is gateset
    assert gateset.with_params(name=gateset.name, unroll_circuit_op=gateset._unroll_circuit_op) is gateset
    gateset_with_params = gateset.with_params(name='new name', unroll_circuit_op=False)
    assert gateset_with_params.name == 'new name'
    assert gateset_with_params._unroll_circuit_op is False