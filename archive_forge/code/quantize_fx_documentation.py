from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import (  # noqa: F401
from .fx.fuse import fuse  # noqa: F401
from .fx.prepare import prepare  # noqa: F401
from .fx.convert import convert
from .backend_config import (  # noqa: F401
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.custom_config import (
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping
 [Internal use only] Convert a model produced by :func:`~torch.ao.quantization.prepare_standalone_module_fx`
    and convert it to a quantized model

    Returns a quantized standalone module, whether input/output is quantized is
    specified by prepare_custom_config, with
    input_quantized_idxs, output_quantized_idxs, please
    see docs for prepare_fx for details
    