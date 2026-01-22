import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
class DecomposeGiven:

    def __init__(self, val):
        self.val = val

    def _decompose_(self):
        return self.val