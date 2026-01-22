import abc
from typing import Any, List, Mapping, Type, Union
from ray.rllib.core.models.base import ENCODER_OUT, STATE_OUT
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import TensorType
BC forward pass during training.

        See the `BCTorchRLModule._forward_exploration` method for
        implementation details.
        