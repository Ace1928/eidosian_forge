import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
@DeveloperAPI
class TorchMultiActionDistribution(TorchDistributionWrapper):
    """Action distribution that operates on multiple, possibly nested actions."""

    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        """Initializes a TorchMultiActionDistribution object.

        Args:
            inputs (torch.Tensor): A single tensor of shape [BATCH, size].
            model (TorchModelV2): The TorchModelV2 object used to produce
                inputs for this distribution.
            child_distributions (any[torch.Tensor]): Any struct
                that contains the child distribution classes to use to
                instantiate the child distributions from `inputs`. This could
                be an already flattened list or a struct according to
                `action_space`.
            input_lens (any[int]): A flat list or a nested struct of input
                split lengths used to split `inputs`.
            action_space (Union[gym.spaces.Dict,gym.spaces.Tuple]): The complex
                and possibly nested action space.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs, model)
        self.action_space_struct = get_base_struct_from_space(action_space)
        self.input_lens = tree.flatten(input_lens)
        flat_child_distributions = tree.flatten(child_distributions)
        split_inputs = torch.split(inputs, self.input_lens, dim=1)
        self.flat_child_distributions = tree.map_structure(lambda dist, input_: dist(input_, model), flat_child_distributions, list(split_inputs))

    @override(ActionDistribution)
    def logp(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if isinstance(x, torch.Tensor):
            split_indices = []
            for dist in self.flat_child_distributions:
                if isinstance(dist, TorchCategorical):
                    split_indices.append(1)
                elif isinstance(dist, TorchMultiCategorical) and dist.action_space is not None:
                    split_indices.append(int(np.prod(dist.action_space.shape)))
                else:
                    sample = dist.sample()
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(sample.size()[1])
            split_x = list(torch.split(x, split_indices, dim=1))
        else:
            split_x = tree.flatten(x)

        def map_(val, dist):
            if isinstance(dist, TorchCategorical):
                val = (torch.squeeze(val, dim=-1) if len(val.shape) > 1 else val).int()
            return dist.logp(val)
        flat_logps = tree.map_structure(map_, split_x, self.flat_child_distributions)
        return functools.reduce(lambda a, b: a + b, flat_logps)

    @override(ActionDistribution)
    def kl(self, other):
        kl_list = [d.kl(o) for d, o in zip(self.flat_child_distributions, other.flat_child_distributions)]
        return functools.reduce(lambda a, b: a + b, kl_list)

    @override(ActionDistribution)
    def entropy(self):
        entropy_list = [d.entropy() for d in self.flat_child_distributions]
        return functools.reduce(lambda a, b: a + b, entropy_list)

    @override(ActionDistribution)
    def sample(self):
        child_distributions = tree.unflatten_as(self.action_space_struct, self.flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(), child_distributions)

    @override(ActionDistribution)
    def deterministic_sample(self):
        child_distributions = tree.unflatten_as(self.action_space_struct, self.flat_child_distributions)
        return tree.map_structure(lambda s: s.deterministic_sample(), child_distributions)

    @override(TorchDistributionWrapper)
    def sampled_action_logp(self):
        p = self.flat_child_distributions[0].sampled_action_logp()
        for c in self.flat_child_distributions[1:]:
            p += c.sampled_action_logp()
        return p

    @override(ActionDistribution)
    def required_model_output_shape(self, action_space, model_config):
        return np.sum(self.input_lens, dtype=np.int32)