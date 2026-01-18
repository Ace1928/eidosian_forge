import functools
import warnings
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration

Note [ONNX operators that are added/updated from opset 7 to opset 8]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
  Expand

Updated operators:
  Min, Max, Sum, Mean: supports multidirectional broadcasting.
  MaxPool: added optional indices output.
  Scan
