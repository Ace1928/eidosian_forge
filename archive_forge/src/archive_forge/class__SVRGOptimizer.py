import mxnet as mx
@mx.optimizer.register
class _SVRGOptimizer(mx.optimizer.Optimizer):
    """_SVRGOptimizer is a wrapper class for two optimizers: _AssignmentOptimizer for accumulating full gradients in the
    KVStore and a default optimizer that is passed in as a parameter in `mod.init_optimizer()`
    The _SVRGOptimizer is designed to be used with SVRGModule only.

    This optimizer accepts the following parameters in addition to those accepted by :class:`.Optimizer`.

    Parameters
    ----------
    default_optimizer: str or Optimizer
        Optimizer passed-in when invoke on mx.mod.init_optimizer in SVRGModule
    """

    def __init__(self, default_optimizer, **kwargs):
        base_param = self._check_params(**kwargs)
        super(_SVRGOptimizer, self).__init__(**base_param)
        if isinstance(default_optimizer, str):
            self.default_opt = mx.optimizer.create(default_optimizer, **kwargs)
        else:
            self.default_opt = default_optimizer
        self.aux_opt = mx.optimizer.create(_AssignmentOptimizer.__name__)

    @staticmethod
    def _check_params(**kwargs):
        """ Reassemble kwargs to identify additional optimizer params for default optimizers. base_params contains
        all the param names in base class Optimizer.

        Parameters
        ----------
        kwargs: dict
            Parameters for the default optimizer

        Returns
        ----------
        default_params: dict
            Optimizer parameters that are defined in base class Optimizer
        """
        optimizer_param = dict(kwargs)
        base_params = ['rescale_grad', 'param_idx2name', 'wd', 'clip_gradient', 'learning_rate', 'lr_scheduler', 'sym', 'begin_num_update', 'multi_precision', 'param_dict']
        default_params = {}
        for key, _ in optimizer_param.items():
            if key in base_params:
                default_params[key] = optimizer_param[key]
        return default_params

    def update(self, index, weight, grad, state):
        """Updates the given parameter using the corresponding gradient and state. If key contains 'full', update with
        `_AssignmentOptimizer` otherwise will use default optimizer.

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
        state : any obj
            The state returned by `create_state()`.
        """
        name = self._check_index(index)
        if 'full' in name:
            self.aux_opt.update(index, weight, grad, state)
        else:
            self.default_opt.update(index, weight, grad, state)

    def create_state(self, index, weight):
        """Creates auxiliary state for a given weight.
        Some optimizers require additional states, e.g. as momentum, in addition
        to gradients in order to update weights. This function creates state
        for a given weight which will be used in `update`. This function is
        called only once for each weight.

        Parameters
        ----------
        index : int
            An unique index to identify the weight.
        weight : NDArray
            The weight.
        Returns
        -------
        state : any obj
            The state associated with the weight.
        """
        name = self._check_index(index)
        if 'full' in name:
            return self.aux_opt.create_state(index, weight)
        else:
            return self.default_opt.create_state(index, weight)

    def _check_index(self, index):
        """Check index in idx2name to get corresponding param_name
        Parameters
        ----------
        index : int or str
            An unique index to identify the weight.
        Returns
        -------
        name : str
            Name of the Module parameter
        """
        if index in self.idx2name.values():
            name = index
        else:
            name = self.idx2name[index]
        return name