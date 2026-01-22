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
@property
def _no_sync_in_gradient_accumulation(self):
    """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
    return not (self.deepspeed or is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled() or is_torch_neuroncore_available())