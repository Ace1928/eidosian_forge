import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
Calculate cubic convolution interpolation coefficients
        ROBERT G. KEYS https://ieeexplore.ieee.org/document/1163711
        Use float to avoid potential issues with integer.
        