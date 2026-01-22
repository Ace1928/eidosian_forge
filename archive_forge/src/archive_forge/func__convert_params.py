from argparse import Namespace
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union
import numpy as np
from torch import Tensor
def _convert_params(params: Optional[Union[Dict[str, Any], Namespace]]) -> Dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params: Target to be converted to a dictionary

    Returns:
        params as a dictionary

    """
    if isinstance(params, Namespace):
        params = vars(params)
    if params is None:
        params = {}
    return params