from __future__ import annotations
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from peft.config import PeftConfig
from peft.utils import (
from .tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge, check_target_module_exists

        Deletes an existing adapter.

        Args:
            adapter_name (`str`): Name of the adapter to be deleted.
        