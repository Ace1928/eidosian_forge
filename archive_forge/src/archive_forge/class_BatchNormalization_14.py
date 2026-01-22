import numpy as np
from onnx.reference.op_run import OpRun
class BatchNormalization_14(OpRun):

    def _run(self, x, scale, bias, mean, var, epsilon=None, momentum=None, training_mode=None):
        if training_mode == 0:
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
            return (res,)
        res, __, _, output_mean, output_var = _batchnorm_training_mode(x, scale, bias, mean, var, momentum, epsilon)
        return (res, output_mean, output_var)