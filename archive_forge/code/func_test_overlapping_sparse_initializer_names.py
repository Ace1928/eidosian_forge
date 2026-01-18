import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_overlapping_sparse_initializer_names(self) -> None:
    """Tests error checking when the name of sparse_initializer entries overlaps"""
    self._test_overlapping_names(sparse_initializer0=['sparse_init0', 'sparse_init1'], sparse_initializer1=['sparse_init1', 'sparse_init2'])