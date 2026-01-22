from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from vllm.model_executor.layers.linear import LinearMethodBase
Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        