import numpy as np
from sklearn.neural_network._stochastic_optimizers import (
from sklearn.utils._testing import assert_array_equal
def test_sgd_optimizer_momentum():
    params = [np.zeros(shape) for shape in shapes]
    lr = 0.1
    rng = np.random.RandomState(0)
    for momentum in np.arange(0.5, 0.9, 0.1):
        optimizer = SGDOptimizer(params, lr, momentum=momentum, nesterov=False)
        velocities = [rng.random_sample(shape) for shape in shapes]
        optimizer.velocities = velocities
        grads = [rng.random_sample(shape) for shape in shapes]
        updates = [momentum * velocity - lr * grad for velocity, grad in zip(velocities, grads)]
        expected = [param + update for param, update in zip(params, updates)]
        optimizer.update_params(params, grads)
        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)