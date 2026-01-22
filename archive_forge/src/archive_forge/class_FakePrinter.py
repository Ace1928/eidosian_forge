import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
class FakePrinter:

    def __init__(self):
        self.text_pretty = ''

    def text(self, to_print):
        self.text_pretty += to_print