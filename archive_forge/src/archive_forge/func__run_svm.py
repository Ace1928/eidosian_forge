import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def _run_svm(self, X, sv, vector_count_, kernel_type_, class_count_, starting_vector_, coefs):
    evals = 0
    kernels_list = []
    for j in range(vector_count_):
        kernels_list.append(self._svm.kernel_dot(X, sv[j], kernel_type_))
    kernels = np.array(kernels_list)
    votes = np.zeros((class_count_,), dtype=X.dtype)
    scores = []
    for i in range(class_count_):
        si_i = starting_vector_[i]
        class_i_sc = self._svm.atts.vectors_per_class[i]
        for j in range(i + 1, class_count_):
            si_j = starting_vector_[j]
            class_j_sc = self._svm.atts.vectors_per_class[j]
            s1 = np.dot(coefs[j - 1, si_i:si_i + class_i_sc], kernels[si_i:si_i + class_i_sc])
            s2 = np.dot(coefs[i, si_j:si_j + class_j_sc], kernels[si_j:si_j + class_j_sc])
            s = self._svm.atts.rho[evals] + s1 + s2
            scores.append(s)
            if s > 0:
                votes[i] += 1
            else:
                votes[j] += 1
            evals += 1
    return (votes, np.array(scores, dtype=X.dtype))