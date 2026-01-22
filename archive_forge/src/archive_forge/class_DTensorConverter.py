import itertools
import sys
from functools import wraps
from typing import (
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import (
from torch.distributed._tensor import (
from torch.distributed._tensor.placement_types import Placement
class DTensorConverter:

    def __init__(self, mesh: DeviceMesh, args: Tuple[object, ...], kwargs: Dict[str, object]) -> None:
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        flatten_args, flatten_args_spec = tree_flatten(args)
        flatten_kwargs, flatten_kwargs_spec = tree_flatten(kwargs)
        self.flatten_args: List[object] = flatten_args
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        self.flatten_kwargs: List[object] = flatten_kwargs
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec
        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))
        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))
        self.sharding_combs: Iterator[Sequence[Placement]] = iter(itertools.product(*choices_for_args))

    def successful(self) -> bool:
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        return not any([t.is_sparse_csr, t.is_sparse, t.is_mkldnn, t.is_quantized, t.is_nested, torch._is_functional_tensor(t), t.is_neg(), t.is_conj(), t.device.type in ('lazy', 'meta')])

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        mesh_size = self.mesh.size()
        sharding_choices: List[Placement] = [Replicate()]
        if arg.dtype != torch.bool:
            sharding_choices = sharding_choices + [Shard(i) for i, s in enumerate(arg.shape) if s > 1 and s % mesh_size == 0]
        return sharding_choices

    def __iter__(self) -> 'DTensorConverter':
        return self

    def __next__(self) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0
            new_args: List[object] = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_args.append(arg)
            new_kwargs: List[object] = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_kwargs.append(arg)
            return (tree_unflatten(new_args, self.flatten_args_spec), tree_unflatten(new_kwargs, self.flatten_kwargs_spec))
        except StopIteration as e:
            raise StopIteration from e

    def to_dist_tensor(self, t: torch.Tensor, mesh: DeviceMesh, placements: List[Placement]) -> torch.Tensor:
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            if self.is_supported_tensor(t):
                self.hit += 1
                if t.ndim == 0:
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    r = distribute_tensor(t, mesh, placements)
                if type(t) is torch.nn.Parameter:
                    r = torch.nn.Parameter(r, requires_grad=r.requires_grad)
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            self.miss += 1
            return t
        else:
            raise RuntimeError(f'Trying to convert to DTensor, but got {type(t)}')