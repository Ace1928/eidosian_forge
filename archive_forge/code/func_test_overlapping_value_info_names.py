import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_overlapping_value_info_names(self) -> None:
    """Tests error checking when the name of value_info entries overlaps"""
    self._test_overlapping_names(value_info0=['vi0', 'vi1'], value_info1=['vi1', 'vi2'])