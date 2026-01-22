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
class SizeInfo:
    """Size info for a file or the repo in bytes.

    Deduped size can't be disabled. So it is always performed.

    Both sparsified and gzipped are optional. They are applied in the following
    order if both are enabled:

        sparsify -> gzip

    Therefore, original >= deduped >= sparsified >= gzipped
    """
    original: int
    deduped: int
    sparsified: int
    gzipped: int