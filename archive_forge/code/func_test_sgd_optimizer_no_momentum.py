import numpy as np
from sklearn.neural_network._stochastic_optimizers import (
from sklearn.utils._testing import assert_array_equal
def test_sgd_optimizer_no_momentum():
    params = [np.zeros(shape) for shape in shapes]
    rng = np.random.RandomState(0)
    for lr in [10 ** i for i in range(-3, 4)]:
        optimizer = SGDOptimizer(params, lr, momentum=0, nesterov=False)
        grads = [rng.random_sample(shape) for shape in shapes]
        expected = [param - lr * grad for param, grad in zip(params, grads)]
        optimizer.update_params(params, grads)
        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)