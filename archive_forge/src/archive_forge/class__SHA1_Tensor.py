import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from .pygit import PyGit
from .sha1_store import SHA1_Store
@dataclass
class _SHA1_Tensor:
    """Representing a tensor using sha1(s) from SHA1 store.

    It can be either a dense one or two sparse one (SST and DST).
    """
    is_dense: bool = True
    dense_sha1: str = ''
    sst_sha1: str = ''
    dst_sha1: str = ''