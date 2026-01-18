from typing import cast
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
test that RigettiQCSSampler can run a basic bell circuit on the QVM and return an accurate
    ``cirq.study.Result``.
    