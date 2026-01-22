from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.fx._equalize import (
from torch.ao.quantization.observer import _is_activation_post_process
def _is_supported(self, module: nn.Module, insert: bool=False) -> bool:
    """Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
    is_supported_type = sum([type(module) is x for x in self.SUPPORTED_MODULES]) > 0
    if insert:
        return is_supported_type
    else:
        has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
        return is_supported_type and has_obs