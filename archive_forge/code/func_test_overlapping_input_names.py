import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_overlapping_input_names(self) -> None:
    """Tests error checking when the name of the inputs overlaps"""
    self._test_overlapping_names(inputs0=['i0', 'i1'], inputs1=['i1', 'i2'])