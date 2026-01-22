import logging
from typing import Optional, Type
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.cql.cql_tf_policy import CQLTFPolicy
from ray.rllib.algorithms.cql.cql_torch_policy import CQLTorchPolicy
from ray.rllib.algorithms.sac.sac import (
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import ResultDict
class CQLConfig(SACConfig):
    """Defines a configuration class from which a CQL can be built.

    .. testcode::
        :skipif: True

        from ray.rllib.algorithms.cql import CQLConfig
        config = CQLConfig().training(gamma=0.9, lr=0.01)
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=4)
        print(config.to_dict())
        # Build a Algorithm object from the config and run 1 training iteration.
        algo = config.build(env="CartPole-v1")
        algo.train()
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CQL)
        self.bc_iters = 20000
        self.temperature = 1.0
        self.num_actions = 10
        self.lagrangian = False
        self.lagrangian_thresh = 5.0
        self.min_q_weight = 5.0
        self.min_sample_timesteps_per_iteration = 0
        self.min_train_timesteps_per_iteration = 100
        self.timesteps_per_iteration = DEPRECATED_VALUE

    @override(SACConfig)
    def training(self, *, bc_iters: Optional[int]=NotProvided, temperature: Optional[float]=NotProvided, num_actions: Optional[int]=NotProvided, lagrangian: Optional[bool]=NotProvided, lagrangian_thresh: Optional[float]=NotProvided, min_q_weight: Optional[float]=NotProvided, **kwargs) -> 'CQLConfig':
        """Sets the training-related configuration.

        Args:
            bc_iters: Number of iterations with Behavior Cloning pretraining.
            temperature: CQL loss temperature.
            num_actions: Number of actions to sample for CQL loss
            lagrangian: Whether to use the Lagrangian for Alpha Prime (in CQL loss).
            lagrangian_thresh: Lagrangian threshold.
            min_q_weight: in Q weight multiplier.

        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if bc_iters is not NotProvided:
            self.bc_iters = bc_iters
        if temperature is not NotProvided:
            self.temperature = temperature
        if num_actions is not NotProvided:
            self.num_actions = num_actions
        if lagrangian is not NotProvided:
            self.lagrangian = lagrangian
        if lagrangian_thresh is not NotProvided:
            self.lagrangian_thresh = lagrangian_thresh
        if min_q_weight is not NotProvided:
            self.min_q_weight = min_q_weight
        return self

    @override(SACConfig)
    def validate(self) -> None:
        if self.timesteps_per_iteration != DEPRECATED_VALUE:
            deprecation_warning(old='timesteps_per_iteration', new='min_train_timesteps_per_iteration', error=True)
        super().validate()
        if self.simple_optimizer is not True and self.framework_str == 'torch':
            self.simple_optimizer = True
        if self.framework_str in ['tf', 'tf2'] and tfp is None:
            logger.warning(f'You need `tensorflow_probability` in order to run CQL! Install it via `pip install tensorflow_probability`. Your tf.__version__={(tf.__version__ if tf else None)}.Trying to import tfp results in the following error:')
            try_import_tfp(error=True)