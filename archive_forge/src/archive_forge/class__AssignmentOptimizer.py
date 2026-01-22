import mxnet as mx
@mx.optimizer.register
class _AssignmentOptimizer(mx.optimizer.Optimizer):
    """_AssignmentOptimizer assigns gradients to weights for SVRGModule's full gradients
    accumulation in the KVStore. It is a helper optimizer that is designed to be used with SVRGModule only.
    """

    def update(self, index, weight, grad, state):
        """Assign the gradients to weight for accumulating full gradients in the KVStore across all devices and workers.

        Parameters
        ----------
        index : int
            The unique index of the parameter into the individual learning
            rates and weight decays. Learning rates and weight decay
            may be set via `set_lr_mult()` and `set_wd_mult()`, respectively.
        weight : NDArray
            The parameter to be updated.
        grad : NDArray
            The gradient of the objective with respect to this parameter.
        state: any obj
            AssignmentOptimizer will not need to be associated with state.
        """
        weight[:] = grad