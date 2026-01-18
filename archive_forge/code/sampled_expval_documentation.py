import numpy as np
from qiskit._accelerate.sampled_exp_val import sampled_expval_float, sampled_expval_complex
from qiskit.exceptions import QiskitError
from .distributions import QuasiDistribution, ProbDistribution
Computes expectation value from a sampled distribution

    Note that passing a raw dict requires bit-string keys.

    Parameters:
        dist (Counts or QuasiDistribution or ProbDistribution or dict): Input sampled distribution
        oper (str or Pauli or PauliOp or PauliSumOp or SparsePauliOp): The operator for
                                                                       the observable

    Returns:
        float: The expectation value
    Raises:
        QiskitError: if the input distribution or operator is an invalid type
    