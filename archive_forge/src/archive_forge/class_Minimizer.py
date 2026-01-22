from __future__ import annotations
import typing
from functools import partial
from collections.abc import Callable
from typing import Protocol
import numpy as np
from qiskit.quantum_info import Operator
from .approximate import ApproximateCircuit, ApproximatingObjective
class Minimizer(Protocol):
    """Callable Protocol for minimizer.

    This interface is based on `SciPy's optimize module
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.

     This protocol defines a callable taking the following parameters:

         fun
             The objective function to minimize.
         x0
             The initial point for the optimization.
         jac
             The gradient of the objective function.
         bounds
             Parameters bounds for the optimization. Note that these might not be supported
             by all optimizers.

     and which returns a SciPy minimization result object.
    """

    def __call__(self, fun: Callable[[np.ndarray], float], x0: np.ndarray, jac: Callable[[np.ndarray], np.ndarray] | None=None, bounds: list[tuple[float, float]] | None=None) -> scipy.optimize.OptimizeResult:
        """Minimize the objective function.

        This interface is based on `SciPy's optimize module <https://docs.scipy.org/doc
        /scipy/reference/generated/scipy.optimize.minimize.html>`__.

        Args:
            fun: The objective function to minimize.
            x0: The initial point for the optimization.
            jac: The gradient of the objective function.
            bounds: Parameters bounds for the optimization. Note that these might not be supported
                by all optimizers.

        Returns:
             The SciPy minimization result object.
        """
        ...