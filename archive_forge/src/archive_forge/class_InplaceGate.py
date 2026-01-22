import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
class InplaceGate(cirq.testing.SingleQubitGate):
    """A gate that modifies the target tensor in place, multiply by -1."""

    def _apply_unitary_(self, args):
        args.target_tensor *= -1.0
        return args.target_tensor