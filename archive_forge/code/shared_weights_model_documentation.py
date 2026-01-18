import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
Example of weight sharing between two different TorchModelV2s.

    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    