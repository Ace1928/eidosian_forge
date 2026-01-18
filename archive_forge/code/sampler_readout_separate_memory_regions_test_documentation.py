from typing import cast, Tuple
import cirq
import pytest
import sympy
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
test that RigettiQCSSampler can properly readout from separate memory
    regions.
    