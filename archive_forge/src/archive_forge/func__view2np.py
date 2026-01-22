import os
import sys
from typing import Dict, Optional, Union
import numpy as np
from safetensors import deserialize, safe_open, serialize, serialize_file
def _view2np(safeview) -> Dict[str, np.ndarray]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v['dtype'])
        arr = np.frombuffer(v['data'], dtype=dtype).reshape(v['shape'])
        result[k] = arr
    return result