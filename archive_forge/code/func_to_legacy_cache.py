from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def to_legacy_cache(self):
    """Dummy function for BC. We have to keep it because otherwise the call in the forward of models will break it"""
    return None