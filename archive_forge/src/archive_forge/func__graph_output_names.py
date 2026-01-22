import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from .functional_utils import _get_mutation_type
from .schemas import (
from .utils import strict_zip
def _graph_output_names(gm):
    output_node = next(iter(reversed(gm.graph.nodes)))
    assert output_node.op == 'output' and len(output_node.args) == 1
    return_args = output_node.args[0]
    return [getattr(return_arg, 'name', None) for return_arg in return_args]