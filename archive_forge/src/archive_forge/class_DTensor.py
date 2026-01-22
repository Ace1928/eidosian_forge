import warnings
from typing import Callable, cast, Optional, Sequence, Tuple
import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed._tensor.dispatch as op_dispatch
import torch.distributed._tensor.random as random
import torch.nn as nn
from torch.distributed._tensor._collective_utils import mesh_broadcast
from torch.distributed._tensor._utils import compute_global_tensor_info
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.random import (
from torch.distributed._tensor.redistribute import (
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
class DTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ['_local_tensor', '_spec']
    _op_dispatcher: op_dispatch.OpDispatcher = op_dispatch.OpDispatcher()

    @staticmethod
    def __new__(cls, local_tensor: torch.Tensor, device_mesh: DeviceMesh, placements: Tuple[Placement, ...], *, shape: torch.Size, dtype: torch.dtype, requires_grad: bool, stride: Tuple[int, ...]) -> 'DTensor':
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).
        Note: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using `DTensor.from_local`, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using `distribute_tensor`.
        """
        if requires_grad != local_tensor.requires_grad:
            warnings.warn("To construct DTensor from torch.Tensor, it's recommended to use local_tensor.detach() and make requires_grad consistent.")
        r = torch.Tensor._make_wrapper_subclass(cls, shape, strides=stride, dtype=dtype, device=local_tensor.device, layout=local_tensor.layout, requires_grad=requires_grad)
        tensor_meta = TensorMeta(shape, stride, dtype)
        r._spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)
        r._local_tensor = local_tensor
        return r

    def __repr__(self):
        return f'DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})'

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return (['_local_tensor'], (self._spec, self.requires_grad))

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec):
        assert flatten_spec is not None, 'Expecting spec to be not None from `__tensor_flatten__` return value!'
        local_tensor = inner_tensors['_local_tensor']
        spec, requires_grad = flatten_spec
        return DTensor(local_tensor, spec.mesh, spec.placements, shape=spec.tensor_meta.shape, dtype=spec.tensor_meta.dtype, requires_grad=requires_grad, stride=spec.tensor_meta.stride)
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return DTensor._op_dispatcher.dispatch(func, args, kwargs or {})

    @staticmethod
    def from_local(local_tensor: torch.Tensor, device_mesh: Optional[DeviceMesh]=None, placements: Optional[Sequence[Placement]]=None, *, run_check: bool=True, shape: Optional[torch.Size]=None, stride: Optional[Tuple[int, ...]]=None) -> 'DTensor':
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the `device_mesh` and `placements` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`. If not
                specified, we will by default replicate the tensor across the
                `device_mesh` from the first rank of each dimension of the `device_mesh`.

        Keyword args:
            run_check (bool, optional): indicate whether to run check across ranks
                to check meta information and data. if have :class:`Replicate` in
                `placements`, the data on first rank of the device mesh dimension
                will be broadcasted to other ranks.
            shape (torch.Size, optional): A List of int which specifies the size of
                DTensor which build on top of `local_tensor`. Note this needs to be
                provided if the shape of `local_tensor` are different across the ranks.
                If not provided, `shape` will be computed assuming the given distributed
                tensor is evenly sharded across ranks.
            stride (tuple, optional): A List of int which specifies the stride of DTensor.
                If not provided, `stride` will be computed assuming the given distributed
                tensor is evenly sharded across ranks.

        Returns:
            A :class:`DTensor` object

        .. note:: `from_local` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
        """
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type
        if device_type != local_tensor.device.type and (not local_tensor.is_meta):
            local_tensor = local_tensor.to(device_type)
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)
        return _FromTorchTensor.apply(local_tensor, device_mesh, tuple(placements), run_check, shape, stride)

    def to_local(self, *, grad_placements: Optional[Sequence[Placement]]=None) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the Tensor returned from this
                function.
                `to_local` converts DTensor to local tensor and the returned local tensor
                might not be used as the original DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original DTensor layout.
                If not specified, we will assume the gradient layout remains the same
                as the original DTensor and use that for gradient computation.

        Returns:
            A :class:`torch.Tensor` or `AsyncCollectiveTensor` object. it represents the
            local tensor on its current rank.

        .. note:: `to_local` is differentiable, the `requires_grad` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.
        """
        if grad_placements is not None and (not isinstance(grad_placements, tuple)):
            grad_placements = tuple(grad_placements)
        return _ToTorchTensor.apply(self, grad_placements, True)

    def redistribute(self, device_mesh: Optional[DeviceMesh]=None, placements: Optional[Sequence[Placement]]=None) -> 'DTensor':
        """
        `redistribute` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from is current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as `device_mesh.ndim`.

        Returns:
            A :class:`DTensor` object

        .. note:: `redistribute` is differentiable.
        """
        device_mesh = device_mesh or self.device_mesh
        if placements is None:
            raise RuntimeError('placements is needed for redistribute!')
        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError('Can not redistribute to _Partial, _Partial is for internal use only!')
            elif isinstance(placement, Shard) and placement.dim < 0:
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)
        if self._spec.placements == placements:
            return self
        return Redistribute.apply(self, device_mesh, placements)

    def full_tensor(self, *, grad_placements: Optional[Sequence[Placement]]=None) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntatic sugar of the following code:

        `dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()`

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.

        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.

        .. note:: `full_tensor` is differentiable.
        """
        redist_res = self.redistribute(placements=[Replicate()] * self.device_mesh.ndim)
        return _ToTorchTensor.apply(redist_res, grad_placements, False)

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: device_mesh is a read-only property, it can not be set.
        """
        return self._spec.mesh

    @property
    def placements(self) -> Sequence[Placement]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: placements is a read-only property, it can not be set.
        """
        return self._spec.placements