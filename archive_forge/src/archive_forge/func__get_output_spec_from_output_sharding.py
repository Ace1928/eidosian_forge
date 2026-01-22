import copy
import operator
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel.style import ColwiseParallel, ParallelStyle
from torch.export import ExportedProgram
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
def _get_output_spec_from_output_sharding(output_sharding: OutputSharding) -> DTensorSpec:
    """
    Util function to extract output spec from output sharding.
    """
    if isinstance(output_sharding.output_spec, DTensorSpec):
        return output_sharding.output_spec
    else:
        assert isinstance(output_sharding.output_spec, Sequence)
        assert output_sharding.output_spec[0] is not None
        output_sharding.output_spec[0].tensor_meta = None
        return output_sharding.output_spec[0]