from typing import Optional
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
Performs a forward pass through this MLP.

        Args:
            input_: The input tensor for the MLP dense stack.
        