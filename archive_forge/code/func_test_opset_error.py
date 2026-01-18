import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_opset_error(self) -> None:
    self.assertRaises(AssertionError, lambda: hub.load(self.name, self.repo, opset=-1))