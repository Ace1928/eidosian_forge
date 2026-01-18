import numpy as np
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch, TensorType
Initializes a NoisyLayer object.

        Args:
            in_size: Input size for Noisy Layer
            out_size: Output size for Noisy Layer
            sigma0: Initialization value for sigma_b (bias noise)
            activation: Non-linear activation for Noisy Layer
        