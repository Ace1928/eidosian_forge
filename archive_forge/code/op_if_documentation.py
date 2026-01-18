from __future__ import annotations
import numpy as np
from onnx.reference.op_run import OpRun
Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Loop).
        The default answer is `False`.
        