import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def triu_reference_implementation(x, k=0):
    return np.triu(x, k)