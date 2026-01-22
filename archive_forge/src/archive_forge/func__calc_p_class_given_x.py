from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
def _calc_p_class_given_x(xs, classes, features, alphas):
    """Calculate conditional probability P(y|x) (PRIVATE).

    y is the class and x is an instance from the training set.
    Return a XSxCLASSES matrix of probabilities.
    """
    prob_yx = np.zeros((len(xs), len(classes)))
    assert len(features) == len(alphas)
    for feature, alpha in zip(features, alphas):
        for (x, y), f in feature.items():
            prob_yx[x][y] += alpha * f
    prob_yx = np.exp(prob_yx)
    for i in range(len(xs)):
        z = sum(prob_yx[i])
        prob_yx[i] = prob_yx[i] / z
    return prob_yx