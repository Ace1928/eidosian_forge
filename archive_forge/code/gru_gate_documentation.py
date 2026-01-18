from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType

        input_shape (torch.Tensor): dimension of the input
        init_bias: Bias added to every input to stabilize training
        