import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_upsample_with_constant_node_9_8(self) -> None:
    self.helper_upsample_with_constant(raw_scale=False)