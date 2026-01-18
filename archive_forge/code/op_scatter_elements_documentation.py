import numpy as np
from onnx.reference.op_run import OpRun
Scatter elements.

    ::
        for 3-dim and axis=0
            output[indices[i][j][k]][j][k] = updates[i][j][k]
        for axis 1
            output[i][indices[i][j][k]][k] = updates[i][j][k]
        and so on.
    