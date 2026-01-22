import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def _run_linear(self, X, coefs, class_count_, kernel_type_):
    scores = []
    for j in range(class_count_):
        d = self._svm.kernel_dot(X, coefs[j], kernel_type_)
        score = self._svm.atts.rho[0] + d
        scores.append(score)
    return np.array(scores, dtype=X.dtype)