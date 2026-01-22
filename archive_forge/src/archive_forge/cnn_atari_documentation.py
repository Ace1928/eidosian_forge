from typing import Optional
from ray.rllib.algorithms.dreamerv3.utils import get_cnn_multiplier
from ray.rllib.utils.framework import try_import_tf
Performs a forward pass through the CNN Atari encoder.

        Args:
            inputs: The image inputs of shape (B, 64, 64, 3).
        