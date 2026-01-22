import numpy as np
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun, RefAttrName
class ConstantCommon(OpRun):

    def _check(self, cst):
        if isinstance(cst, tuple):
            raise TypeError(f'Unexpected type {type(cst)} for a constant.')
        return cst