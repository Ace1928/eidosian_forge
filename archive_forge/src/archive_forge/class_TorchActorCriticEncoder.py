from typing import Optional
import tree
from ray.rllib.core.models.base import (
from ray.rllib.core.models.base import Model, tokenize
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchMLP, TorchCNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
class TorchActorCriticEncoder(TorchModel, ActorCriticEncoder):
    """An actor-critic encoder for torch."""
    framework = 'torch'

    def __init__(self, config: ActorCriticEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        ActorCriticEncoder.__init__(self, config)