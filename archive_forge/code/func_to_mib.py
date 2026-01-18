from typing import Dict, Any, Optional, IO
import sys
from spacy import Language
from thinc.backends import context_pools
from thinc.util import has_cupy_gpu
from .util import LoggerT
def to_mib(bytes: int) -> float:
    return bytes / 1024.0 ** 2