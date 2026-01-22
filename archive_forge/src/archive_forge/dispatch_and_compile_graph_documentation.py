from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import lazy_format_graph_code
from torch._logging import getArtifactLogger
from torch.fx.experimental.proxy_tensor import make_fx
from .functional_utils import assert_functional_graph
from .schemas import AOTConfig, SubclassMeta, ViewAndMutationMeta
from .traced_function_transforms import (

This module dispatches the graphs to either the forward-only or joint compilation
pathways, taking into account the AOTConfig and the collected ViewAndMutationMetadata.
