from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from typing import Optional
Initializes a SkipConnection nn Module object.

        Args:
            layer (nn.Module): Any layer processing inputs.
            fan_in_layer (Optional[nn.Module]): An optional
                layer taking two inputs: The original input and the output
                of `layer`.
        