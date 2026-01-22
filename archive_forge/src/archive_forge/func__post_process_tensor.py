import warnings
from scipy.linalg import sqrtm
import numpy as np
import pennylane as qml
def _post_process_tensor(self, tensor_raw_results, tensor_dirs):
    """Post process the corresponding tape results to get the metric tensor estimation.

        Args:
            tensor_raw_results list[np.array]: list of the four perturbed qnode results to compute
            the estimated metric tensor
            tensor_dirs list[np.array]: list of the two perturbation directions used to compute
            the metric tensor estimation. Perturbations on the different input parameters have
            been concatenated

        Returns:
            np.array: estimated Fubini-Study metric tensor
        """
    tensor_raw_results = [result.squeeze() for result in tensor_raw_results]
    tensor_finite_diff = tensor_raw_results[0][0] - tensor_raw_results[1][0] - tensor_raw_results[2][0] + tensor_raw_results[3][0]
    return -(np.tensordot(tensor_dirs[0], tensor_dirs[1], axes=0) + np.tensordot(tensor_dirs[1], tensor_dirs[0], axes=0)) * tensor_finite_diff / (8 * self.finite_diff_step ** 2)