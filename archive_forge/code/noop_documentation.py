from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
Trivial model that just returns the obs flattened.

    This is the model used if use_state_preprocessor=False.