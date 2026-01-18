import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def to_sanitized_dict(self) -> Dict[str, Any]:
    """
        Sanitized serialization to use with TensorBoard’s hparams
        """
    d = self.to_dict()
    d = {**d, **{'train_batch_size': self.train_batch_size, 'eval_batch_size': self.eval_batch_size}}
    valid_types = [bool, int, float, str]
    if is_torch_available():
        valid_types.append(torch.Tensor)
    return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}