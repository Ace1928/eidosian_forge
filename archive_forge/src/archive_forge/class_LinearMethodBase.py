from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.utils import (
from vllm.model_executor.utils import set_weight_attrs
from vllm.logger import init_logger
class LinearMethodBase(ABC):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, input_size_per_partition: int, output_size_per_partition: int, input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        """Create weights for a linear layer."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(self, weights: Dict[str, torch.Tensor], x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Apply the weights to the input tensor."""
        raise NotImplementedError