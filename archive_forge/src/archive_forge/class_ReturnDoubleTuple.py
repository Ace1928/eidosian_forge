import numpy as np
import pytest
import sympy
import cirq
class ReturnDoubleTuple:

    def _circuit_diagram_info_(self, args):
        return ('Single', 'Double')