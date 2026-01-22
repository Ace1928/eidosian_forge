import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]