import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
class SVMClassifier(OpRunAiOnnxMl):

    def _run_linear(self, X, coefs, class_count_, kernel_type_):
        scores = []
        for j in range(class_count_):
            d = self._svm.kernel_dot(X, coefs[j], kernel_type_)
            score = self._svm.atts.rho[0] + d
            scores.append(score)
        return np.array(scores, dtype=X.dtype)

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

    def _probabilities(self, scores, class_count_):
        probsp2 = np.zeros((class_count_, class_count_), dtype=scores.dtype)
        index = 0
        for i in range(class_count_):
            for j in range(i + 1, class_count_):
                val1 = sigmoid_probability(scores[index], self._svm.atts.prob_a[index], self._svm.atts.prob_b[index])
                val2 = max(val1, 1e-07)
                val2 = min(val2, 1 - 1e-07)
                probsp2[i, j] = val2
                probsp2[j, i] = 1 - val2
                index += 1
        return multiclass_probability(class_count_, probsp2)

    def _compute_final_scores(self, votes, scores, weights_are_all_positive_, has_proba, classlabels_ints):
        max_weight = 0
        if votes is not None and len(votes) > 0:
            max_class = np.argmax(votes)
            max_weight = votes[max_class]
        else:
            max_class = np.argmax(scores)
            max_weight = scores[max_class]
        write_additional_scores = -1
        if self._svm.atts.rho.size == 1:
            label, write_additional_scores = set_score_svm(max_weight, max_class, has_proba, weights_are_all_positive_, classlabels_ints, 1, 0)
        elif classlabels_ints is not None and len(classlabels_ints) > 0:
            label = classlabels_ints[max_class]
        else:
            label = max_class
        new_scores = write_scores(scores.size, scores, self._svm.atts.post_transform, write_additional_scores)
        return (label, new_scores)

    def _run(self, X, classlabels_ints=None, classlabels_strings=None, coefficients=None, kernel_params=None, kernel_type=None, post_transform=None, prob_a=None, prob_b=None, rho=None, support_vectors=None, vectors_per_class=None):
        svm = SVMCommon(coefficients=coefficients, kernel_params=kernel_params, kernel_type=kernel_type, post_transform=post_transform, prob_a=prob_a, prob_b=prob_b, rho=rho, support_vectors=support_vectors, vectors_per_class=vectors_per_class)
        self._svm = svm
        vector_count_ = 0
        class_count_ = max(len(classlabels_ints or classlabels_strings or []), 1)
        starting_vector_ = []
        if svm.atts.vectors_per_class is not None:
            for vc in svm.atts.vectors_per_class:
                starting_vector_.append(vector_count_)
                vector_count_ += vc
        if vector_count_ > 0:
            mode = 'SVM_SVC'
            sv = svm.atts.support_vectors.reshape((vector_count_, -1))
            kernel_type_ = svm.atts.kernel_type
            coefs = svm.atts.coefficients.reshape((-1, vector_count_))
        else:
            mode = 'SVM_LINEAR'
            kernel_type_ = 'LINEAR'
            coefs = svm.atts.coefficients.reshape((class_count_, -1))
        weights_are_all_positive_ = min(svm.atts.coefficients) >= 0
        if vector_count_ == 0 and mode == 'SVM_LINEAR':
            res = np.empty((X.shape[0], class_count_), dtype=X.dtype)
            for n in range(X.shape[0]):
                scores = self._run_linear(X[n], coefs, class_count_, kernel_type_)
                res[n, :] = scores
            votes = None
        else:
            res = np.empty((X.shape[0], class_count_ * (class_count_ - 1) // 2), dtype=X.dtype)
            votes = np.empty((X.shape[0], class_count_), dtype=X.dtype)
            for n in range(X.shape[0]):
                vote, scores = self._run_svm(X[n], sv, vector_count_, kernel_type_, class_count_, starting_vector_, coefs)
                res[n, :] = scores
                votes[n, :] = vote
        if svm.atts.prob_a is not None and len(svm.atts.prob_a) > 0 and (mode == 'SVM_SVC'):
            scores = np.empty((res.shape[0], class_count_), dtype=X.dtype)
            for n in range(scores.shape[0]):
                s = self._probabilities(res[n], class_count_)
                scores[n, :] = s
            has_proba = True
        else:
            scores = res
            has_proba = False
        final_scores = None
        labels = []
        for n in range(scores.shape[0]):
            label, new_scores = self._compute_final_scores(None if votes is None else votes[n], scores[n], weights_are_all_positive_, has_proba, classlabels_ints)
            if final_scores is None:
                final_scores = np.empty((X.shape[0], new_scores.size), dtype=X.dtype)
            final_scores[n, :] = new_scores
            labels.append(label)
        if classlabels_strings is not None and len(classlabels_strings) > 0:
            return (np.array([classlabels_strings[i] for i in labels]), final_scores)
        return (np.array(labels, dtype=np.int64), final_scores)