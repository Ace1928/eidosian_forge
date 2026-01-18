import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_typecheck(self) -> None:
    defs.get_schema('Conv')