from typing import Optional
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.reward_predictor_layer import (
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
def update_ema(self) -> None:
    """Updates the EMA-copy of the critic according to the update formula:

        ema_net=(`ema_decay`*ema_net) + (1.0-`ema_decay`)*critic_net
        """
    vars = self.mlp.trainable_variables + self.return_layer.trainable_variables
    vars_ema = self.mlp_ema.variables + self.return_layer_ema.variables
    assert len(vars) == len(vars_ema) and len(vars) > 0
    for var, var_ema in zip(vars, vars_ema):
        var_ema.assign(self.ema_decay * var_ema + (1.0 - self.ema_decay) * var)