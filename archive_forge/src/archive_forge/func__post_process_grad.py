import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _post_process_grad(self, grad_raw_results, grad_dirs):
    """Post process the gradient tape results to get the SPSA gradient estimation.

        Args:
            grad_raw_results list[np.array]: list of the two qnode results with input parameters
            perturbed along the ``grad_dirs`` directions
            grad_dirs list[np.array]: list of perturbation arrays along which the SPSA
            gradients are estimated

        Returns:
            list[np.array]: list of gradient arrays. Each gradient array' dimension matches
            the shape of the corresponding input parameter
        """
    loss_plus, loss_minus = grad_raw_results
    return [(loss_plus - loss_minus) / (2 * self.finite_diff_step) * grad_dir for grad_dir in grad_dirs]