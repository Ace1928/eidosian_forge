import math
from typing import List, Optional, Tuple
import torch
def _get_weight_init_gains(weight_init_scale_strategy: Optional[str], num_layers: int) -> List[Optional[float]]:
    if weight_init_scale_strategy is None:
        return [None for _ in range(num_layers)]
    elif weight_init_scale_strategy == 'depthwise':
        return [1.0 / math.sqrt(layer_idx + 1) for layer_idx in range(num_layers)]
    elif weight_init_scale_strategy == 'constant':
        return [1.0 / math.sqrt(2) for layer_idx in range(num_layers)]
    else:
        raise ValueError(f'Unsupported weight_init_scale_strategy value {weight_init_scale_strategy}')