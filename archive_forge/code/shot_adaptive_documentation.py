from copy import copy
import numpy as np
from scipy.stats import multinomial
import pennylane as qml
from .gradient_descent import GradientDescentOptimizer
Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        The objective function will be evaluated using the maximum number of shots
        across all parameters as determined by the optimizer during the
        optimization step.

        .. warning::

            Unlike other gradient descent optimizers, the objective function will be evaluated
            **separately** to the gradient computation, and will result in extra
            device evaluations.

        Args:
            objective_fn (function): the objective function for optimization
            *args : variable length argument list for objective function
            **kwargs : variable length of keyword arguments for the objective function

        Returns:
            tuple[list [array], float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
            If single arg is provided, list [array] is replaced by array.
        