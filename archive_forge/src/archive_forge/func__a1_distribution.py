from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def _a1_distribution(self):
    BATCH = self.inputs.shape[0]
    zeros = torch.zeros((BATCH, 1)).to(self.inputs.device)
    a1_logits, _ = self.model.action_module(self.inputs, zeros)
    a1_dist = TorchCategorical(a1_logits)
    return a1_dist