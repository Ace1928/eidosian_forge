from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny

        Creates a QConfigMultiMapping from a list of QConfigMappings
        