import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
@staticmethod
def third_party_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party')