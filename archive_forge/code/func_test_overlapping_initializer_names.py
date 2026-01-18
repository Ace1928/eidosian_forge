import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_overlapping_initializer_names(self) -> None:
    """Tests error checking when the name of initializer entries overlaps"""
    self._test_overlapping_names(initializer0=['init0', 'init1'], initializer1=['init1', 'init2'])