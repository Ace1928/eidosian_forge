import math
from typing import Optional, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation.complex_phase_oracle import ComplexPhaseOracle
from cirq_ft.infra import bit_tools
from cirq_ft.infra import testing as cq_testing
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_phase_oracle_consistent_protocols():
    bitsize, arctan_bitsize = (3, 5)
    gate = ComplexPhaseOracle(ExampleSelect(bitsize, 1), arctan_bitsize)
    expected_symbols = ('@',) + ('ROTy',) * bitsize
    assert cirq.circuit_diagram_info(gate).wire_symbols == expected_symbols