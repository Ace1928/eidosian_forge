import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_overlapping_output_names(self) -> None:
    """Tests error checking when the name of the output overlaps"""
    self._test_overlapping_names(outputs0=['o0', 'o1'], outputs1=['o1', 'o2'])