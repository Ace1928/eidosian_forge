from argparse import Namespace
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union
import numpy as np
from torch import Tensor
def _add_prefix(metrics: Mapping[str, Union[Tensor, float]], prefix: str, separator: str) -> Mapping[str, Union[Tensor, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics
    return {f'{prefix}{separator}{k}': v for k, v in metrics.items()}