from functools import lru_cache
from itertools import chain
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.device_mesh import DeviceMesh
def _wrap_output_spec_tensor_meta(self, op: OpOverload, output_spec: OutputSpecType, output_tensor_meta: Union[None, TensorMeta, List[TensorMeta], Tuple[TensorMeta, ...]]) -> None:
    """
        Wrap the output_spec with the tensor metadata from the output.
        """
    if isinstance(output_spec, DTensorSpec):
        if not isinstance(output_tensor_meta, TensorMeta):
            if not isinstance(output_tensor_meta, (tuple, list)):
                raise ValueError('ShardingPropagator error: output does not have an associated TensorMeta')
            raise ValueError(f'For the op {op.name()}, `output_spec` has 1 output which does not equal the number of op outputs: {len(output_tensor_meta)}.')
        output_spec.tensor_meta = output_tensor_meta
    elif isinstance(output_spec, (tuple, list)):
        if not isinstance(output_tensor_meta, (tuple, list)) or len(output_spec) != len(output_tensor_meta):
            raise ValueError(f'For the op {op.name()}, `output_spec` has {len(output_spec)} outputs which does not equal the number of op outputs {_length(output_tensor_meta)}.')
        for i, spec in enumerate(output_spec):
            if isinstance(spec, DTensorSpec):
                output_tensor_meta_i = output_tensor_meta[i]
                if not isinstance(output_tensor_meta_i, TensorMeta):
                    raise ValueError(f'ShardingPropagator error: output {i} does not have an associated TensorMeta')
                spec.tensor_meta = output_tensor_meta_i