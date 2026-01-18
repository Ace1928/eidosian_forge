from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as TFFCNet
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.framework import try_import_tf, try_import_torch
Example of interpreting repeated observations.