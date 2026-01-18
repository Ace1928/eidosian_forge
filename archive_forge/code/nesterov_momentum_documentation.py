from pennylane._grad import grad as get_gradient
from .momentum import MomentumOptimizer
Compute gradient of the objective function at at the shifted point :math:`(x -
        m\times\text{accumulation})` and return it along with the objective function forward pass
        (if available).

        Args:
            objective_fn (function): the objective function for optimization.
            args (tuple): tuple of NumPy arrays containing the current values for the
                objection function.
            kwargs (dict): keyword arguments for the objective function.
            grad_fn (function): optional gradient function of the objective function with respect to
                the variables ``x``. If ``None``, the gradient function is computed automatically.
                Must return the same shape of tuple [array] as the autograd derivative.

        Returns:
            tuple [array]: the NumPy array containing the gradient :math:`\nabla f(x^{(t)})` and the
            objective function output. If ``grad_fn`` is provided, the objective function
            will not be evaluted and instead ``None`` will be returned.
        