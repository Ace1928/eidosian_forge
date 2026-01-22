from copy import copy
import numpy as np
from scipy.stats import multinomial
import pennylane as qml
from .gradient_descent import GradientDescentOptimizer
def _single_shot_qnode_gradients(self, qnode, args, kwargs):
    """Compute the single shot gradients of a QNode."""
    self.check_device(qnode.device)
    qnode.construct(args, kwargs)
    tape = qnode.tape
    [expval] = tape.measurements
    coeffs, observables = expval.obs.terms() if isinstance(expval.obs, qml.Hamiltonian) else ([1.0], [expval.obs])
    if self.lipschitz is None:
        self.check_learning_rate(coeffs)
    if self.term_sampling == 'weighted_random_sampling':
        return self.qnode_weighted_random_sampling(qnode, coeffs, observables, self.max_shots, self.trainable_args, *args, **kwargs)
    if self.term_sampling is not None:
        raise ValueError(f"Unknown Hamiltonian term sampling method {self.term_sampling}. Only term_sampling='weighted_random_sampling' and term_sampling=None currently supported.")
    new_shots = [(1, int(self.max_shots))]

    def cost(*args, **kwargs):
        return qml.math.stack(qnode(*args, **kwargs, shots=new_shots))
    grads = [qml.jacobian(cost, argnum=i)(*args, **kwargs) for i in self.trainable_args]
    return grads