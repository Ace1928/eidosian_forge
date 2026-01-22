from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class ExampleComposite:

    def _decompose_(self):
        return ()