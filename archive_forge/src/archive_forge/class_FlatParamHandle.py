import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
class FlatParamHandle:
    """
    This handle manages a flat parameter (:class:`FlatParameter`). This
    includes sharding and view management.

    Args:
        params (Sequence[nn.Parameter]): The parameters to flatten into the
            flat parameter.
        fully_sharded_module (nn.Module): See [Note: Fully Sharded Module].
        device (torch.device): The compute and communication device, which
            should be a non-CPU device. We refer to it as the compute device.
        sharding_strategy (ShardingStrategy): Sharding strategy to apply to
            this handle's ``FlatParameter``.
        offload_params (bool): Whether to offload the handle's
            ``FlatParameter`` to CPU.
        mp_param_dtype (Optional[torch.dtype]): Parameter mixed precision
            setting passed to the FSDP constructor.
        mp_reduce_dtype (Optional[torch.dtype]): Gradient reduction mixed
            precision setting passed to the FSDP constructor.
        keep_low_precision_grads (bool): Whether to keep gradients in low
            precision.
        use_orig_params (bool): If ``True``, then FSDP preserves the original
            parameter variables and returns them from ``named_parameters()``
            (e.g. to support different optimizer hyperparameters within one
            :class:`FlatParameter`). If ``False``, then FSDP reconstructs the
            parameters every iteration and returns the :class:`FlatParameter` s
            from ``named_parameters()``.
    """

    def __init__(self, params: Sequence[Union[nn.Parameter, Tensor]], fully_sharded_module: nn.Module, device: torch.device, sharding_strategy: HandleShardingStrategy, offload_params: bool, mp_param_dtype: Optional[torch.dtype], mp_reduce_dtype: Optional[torch.dtype], keep_low_precision_grads: bool, process_group: dist.ProcessGroup, use_orig_params: bool):
        super().__init__()
        params = list(params)
        if len(params) == 0:
            raise ValueError(f'Cannot construct a {self.__class__.__name__} with an empty parameter list')
        self._init_setattr_fns()
        self._skip_writeback_check = os.environ.get(_FSDP_SKIP_WRITEBACK_CHECK, '') == '1'
        self._use_full_prec_in_eval = os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, '') == '1'
        if self._skip_writeback_check:
            _warn_skip_writeback_check(log, f'Since {_FSDP_SKIP_WRITEBACK_CHECK}=1, FSDP will not check for parameter or gradient writeback. Changing parameter or gradient storages may lead to silent correctness errors.')
        align_addresses = use_orig_params
        self._init_get_unflat_views_fn(align_addresses)
        self.device = device
        self._device_handle = _FSDPDeviceHandle.from_device(self.device)
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        self._sharding_strategy = sharding_strategy
        self._offload_params = offload_params
        self._use_orig_params = use_orig_params
        self._keep_low_precision_grads = keep_low_precision_grads
        self._training_state = HandleTrainingState.IDLE
        self._debug_level = dist.get_debug_level()
        self._fully_sharded_module = fully_sharded_module
        self._unsharded_flat_param_for_skipped_views: Optional[Tensor] = None
        self._handle_index: Optional[int] = None
        self._pre_forward_order_index: Optional[int] = None
        self._post_forward_index: Optional[int] = None
        self._needs_pre_forward_unshard = False
        self._needs_pre_backward_unshard = False
        self._prefetched = False
        self._orig_param_dtype = params[0].dtype
        self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
        assert self._fwd_bwd_param_dtype is not None
        self._aligned_numel = _get_aligned_numel(unsharded_dtype=self._fwd_bwd_param_dtype) if align_addresses else 0
        self._init_flat_param_and_metadata(params, fully_sharded_module, self._aligned_numel, use_orig_params)
        self._use_unsharded_views(as_params=False)

    def _init_setattr_fns(self):
        use_unsafe_setattr = os.environ.get(_FSDP_USE_UNSAFE_SETATTR, '') == '1'
        self._setattr_tensor: Callable[[nn.Module, str, Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

    def _init_get_unflat_views_fn(self, align_addresses: bool):
        self._get_unflat_views = self._get_unflat_views_aligned if align_addresses else self._get_unflat_views_unaligned

    def _init_flat_param_and_metadata(self, params: List[Union[Tensor, nn.Parameter]], module: nn.Module, aligned_numel: int, use_orig_params: bool) -> None:
        """
        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
        if len(params) == 0:
            raise ValueError('Expects non-empty `params`')
        if aligned_numel < 0:
            raise ValueError(f'Expects non-negative `aligned_numel` but got {aligned_numel}')
        dtype, flat_param_requires_grad, device = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        fqns: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[Union[Tensor, nn.Parameter], Tuple[nn.Module, str, str]] = {}
        params_to_flatten: List[Union[Tensor, nn.Parameter]] = []
        shared_params: List[Union[Tensor, nn.Parameter]] = []
        param_extensions: List[Any] = []
        is_padding_mask: List[bool] = []
        total_numel = total_numel_without_padding = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(submodule, recurse=False):
                if param not in params_set:
                    continue
                if param in shared_param_memo:
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[param]
                    shared_params.append(param)
                    shared_param_infos.append(SharedParamInfo(param_name, submodule, submodule_name, prim_param_name, prim_module, prim_module_name))
                else:
                    if aligned_numel > 0:
                        numel_to_pad = aligned_numel - total_numel % aligned_numel
                        if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                            padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
                            params_to_flatten.append(padding_tensor)
                            is_padding_mask.append(True)
                            numels.append(numel_to_pad)
                            total_numel += numel_to_pad
                    transform_t, extension = _ext_pre_flatten_transform(param)
                    param = cast(nn.Parameter, transform_t)
                    param_extensions.append(extension)
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    is_padding_mask.append(False)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    fqn = submodule_name + '.' + param_name if submodule_name else param_name
                    fqns.append(fqn)
                    total_numel += param.numel()
                    total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(f"`params` were not found in `module`'s treeparams: {params}\nmodule: {module}")
        if self.rank == 0 and aligned_numel > 0 and (total_numel != total_numel_without_padding):
            log.info('FSDP FlatParameter address alignment created %s numel of padding (%s vs. %s)', total_numel - total_numel_without_padding, total_numel, total_numel_without_padding)
        if aligned_numel > 0:
            numel_to_pad = self.world_size - total_numel % self.world_size
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                if self.rank == 0:
                    log.info('FSDP FlatParameter world size divisibility created %s numel of padding', numel_to_pad)
                padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
                params_to_flatten.append(padding_tensor)
                is_padding_mask.append(True)
                numels.append(numel_to_pad)
                total_numel += numel_to_pad
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(params_to_flatten, aligned_numel=0, requires_grad=flat_param_requires_grad)
        FlatParameter._init_metadata(self.flat_param, param_infos, numels, shapes, fqns, shared_param_infos, param_extensions, _convert_to_params(params_to_flatten) if use_orig_params else None, _convert_to_params(shared_params) if use_orig_params else None, is_padding_mask)

    def _validate_tensors_to_flatten(self, tensors: List[Union[Tensor, nn.Parameter]]) -> Tuple:
        """
        Validates the tensors to flatten and returns any necessary metadata.
        """
        dtype: Optional[torch.dtype] = None
        flat_param_requires_grad: Optional[bool] = None
        device: Optional[torch.device] = None
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError('Cannot flatten a `FlatParameter`')
            if dtype is None and (not tensor.is_floating_point()):
                raise ValueError('Cannot flatten integer dtype tensors')
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f'Must flatten tensors with uniform dtype but got {dtype} and {tensor.dtype}')
            if not self._use_orig_params and flat_param_requires_grad is not None and (tensor.requires_grad != flat_param_requires_grad):
                raise ValueError('Must flatten tensors with uniform `requires_grad` when `use_orig_params=False`')
            if device is not None and tensor.device != device:
                raise ValueError(f'Must flatten tensors on the same device but got both {device} and {tensor.device}')
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        assert flat_param_requires_grad is not None, 'Requires non-empty `tensors` list'
        return (dtype, flat_param_requires_grad, device)

    def flatten_tensors(self, tensors: List[Tensor], aligned_numel: int) -> Tensor:
        """
        Flattens ``tensors`` into a single flat tensor optionally including
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        if len(tensors) == 0:
            raise ValueError('Expects non-empty `tensors`')
        if aligned_numel < 0:
            raise ValueError(f'Expects non-negative `aligned_numel` but got {aligned_numel}')
        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        flat_tensors: List[Tensor] = []
        if aligned_numel > 0:
            total_numel = 0
            for tensor in tensors:
                numel_to_pad = aligned_numel - total_numel % aligned_numel
                if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                    padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
                    flat_tensors.append(padding_tensor)
                    total_numel += numel_to_pad
                flat_tensors.append(torch.flatten(_detach_if_needed(tensor)))
                total_numel += tensor.numel()
            numel_to_pad = self.world_size - total_numel % self.world_size
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                padding_tensor = _construct_padding_tensor(numel_to_pad, dtype, False, device)
                flat_tensors.append(padding_tensor)
                total_numel += numel_to_pad
        else:
            flat_tensors = [torch.flatten(_detach_if_needed(tensor)) for tensor in tensors]
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(self, tensors: List[Tensor], aligned_numel: int, requires_grad: bool) -> FlatParameter:
        flat_param_data = self.flatten_tensors(tensors, aligned_numel)
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_param_reduce_dtypes(self, mp_param_dtype: Optional[torch.dtype], mp_reduce_dtype: Optional[torch.dtype]) -> None:
        """
        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
        self._low_prec_param_dtype_specified = mp_param_dtype is not None
        self._low_prec_reduce_dtype_specified = mp_reduce_dtype is not None
        if self._low_prec_param_dtype_specified and (not self._low_prec_reduce_dtype_specified):
            self._fwd_bwd_param_dtype = mp_param_dtype
            self._reduce_dtype = self._fwd_bwd_param_dtype
        else:
            self._fwd_bwd_param_dtype = mp_param_dtype or self._orig_param_dtype
            self._reduce_dtype = mp_reduce_dtype or self._orig_param_dtype
        assert self._fwd_bwd_param_dtype is not None
        assert self._reduce_dtype is not None

    @torch.no_grad()
    def shard(self):
        """
        Shards the handle's ``FlatParameter``. This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(flat_param.storage_offset() == 0, 'The `FlatParameter` is not the sole occupant of its storage')
            orig_storage = flat_param._typed_storage()
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(flat_param, self.rank, self.world_size)
            flat_param.set_(sharded_flat_param)
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
            if orig_storage._size() > 0:
                orig_storage._resize_(0)
        if self._use_orig_params:
            self._use_sharded_views()

    def _init_shard_metadata(self, numel_padded: int, unsharded_start_idx: int, unsharded_end_idx: int) -> None:
        """
        Initializes shard-related metadata for this rank's shard of the flat
        parameter: ``_sharded_size``, ``_shard_param_infos``, and
        ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()
        sharded_flat_param_numel = flat_param.numel()
        _p_assert(unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx, f'unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}')
        _p_assert(numel_padded <= sharded_flat_param_numel, f'numel_padded: {numel_padded} sharded_flat_param_numel: {sharded_flat_param_numel}')
        shard_param_infos = self._get_shard_metadata(unsharded_start_idx, unsharded_end_idx)
        assert len(shard_param_infos) == flat_param._num_params, f'Expects length {flat_param._num_params} but got {len(shard_param_infos)}'
        flat_param._shard_param_infos = shard_param_infos
        flat_param._shard_numel_padded = numel_padded

    def _get_shard_metadata(self, unsharded_start_idx: int, unsharded_end_idx: int) -> Tuple[_ShardParamInfo, ...]:
        """
        Computes the shard metadata based on ``unsharded_start_idx`` and
        ``unsharded_end_idx`` (inclusive), which give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        assert len(flat_param_offsets) == len(self.flat_param._numels_with_padding), f'Expected {len(self.flat_param._numels_with_padding)} but got {len(flat_param_offsets)}'
        shard_param_infos: List[_ShardParamInfo] = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        for i, ((unsharded_param_start_idx, unsharded_param_end_idx), is_padding) in enumerate(zip(flat_param_offsets, self.flat_param._is_padding_mask)):
            if is_padding:
                continue
            in_sharded_flat_param = unsharded_start_idx <= unsharded_param_end_idx and unsharded_end_idx >= unsharded_param_start_idx
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = unsharded_start_idx - unsharded_param_start_idx
                    offset_in_shard = 0
                assert offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel, f'Invalid `offset_in_shard` of {offset_in_shard} for sharded flat parameter with {sharded_flat_param_numel} numel'
                intra_param_end_idx = min(unsharded_param_end_idx, unsharded_end_idx) - unsharded_param_start_idx
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(True, offset_in_shard, numel_in_shard, intra_param_start_idx, intra_param_end_idx)
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    @staticmethod
    def _get_unpadded_shard(tensor: Tensor, rank: int, world_size: int) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` without any padding for the given
        ``rank`` and ``world_size`` and the numel to pad for that shard.

        If ``tensor`` is already flattened or may be viewed in the flattened
        shape (which is true in the expected usage), then this method does not
        allocate any new tensor memory.
        """
        chunks = torch.flatten(tensor).chunk(world_size)
        if len(chunks) < rank + 1:
            chunk = chunks[0].new_empty(0)
        else:
            chunk = chunks[rank]
        numel_to_pad = chunks[0].numel() - chunk.numel()
        assert numel_to_pad >= 0, "Chunk's size should be at most the first chunk's size"
        return (chunk, numel_to_pad)

    @staticmethod
    def _get_shard(tensor: Tensor, rank: int, world_size: int) -> Tuple[Tensor, int]:
        """
        Returns the shard of ``tensor`` with padding for the given ``rank`` and
        ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
        chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(tensor, rank, world_size)
        shard = chunk.clone()
        if numel_to_pad > 0:
            shard = F.pad(shard, [0, numel_to_pad])
        return (shard, numel_to_pad)

    @staticmethod
    def _get_sharded_size(tensor: Tensor, rank: int, world_size: int) -> torch.Size:
        """
        Returns the shape of ``tensor`` after sharding including padding. This
        requires ``tensor`` to have 1D shape and ensures that the returned
        shape is 1D.
        """
        assert len(tensor.shape) == 1, f'{tensor.shape}'
        unpadded_sharded_tensor, numel_to_pad = FlatParamHandle._get_unpadded_shard(tensor, rank, world_size)
        unpadded_sharded_size = unpadded_sharded_tensor.size()
        assert len(unpadded_sharded_size) == 1, f'{unpadded_sharded_size}'
        return torch.Size([unpadded_sharded_size[0] + numel_to_pad])

    def _get_flat_param_offsets(self) -> List[Tuple[int, int]]:
        """
        Returns [start, end] offsets of each original parameter's flattened
        data in the unsharded flat parameter (without padding).
        NOTE: The returned list includes elements for alignment padding.
        """
        cumulative_sum = list(accumulate(self.flat_param._numels_with_padding))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]
        param_offsets = list(zip(starts, ends))
        return param_offsets

    @no_type_check
    def shard_metadata(self) -> FlatParamShardMetadata:
        """
        Returns shard-related metadata specific to this rank's shard of the
        flat parameter.
        NOTE: The returned tuple does not include elements for alignment
        padding but does account for the padding.
        """
        fqns_list = []
        shapes_list = []
        numels_list = []
        shard_param_offsets = []
        for fqn, shape, numel, shard_param_info in zip(self.flat_param._fqns, self.flat_param._shapes, self.flat_param._numels, self.flat_param._shard_param_infos):
            if not shard_param_info.in_shard:
                continue
            fqns_list.append(fqn)
            shapes_list.append(shape)
            numels_list.append(numel)
            shard_param_offsets.append((shard_param_info.intra_param_start_idx, shard_param_info.intra_param_end_idx))
        return FlatParamShardMetadata(tuple(fqns_list), tuple(shapes_list), tuple(numels_list), shard_param_offsets)

    @no_type_check
    @torch.no_grad()
    def init_flat_param_attributes(self) -> None:
        """
        This initializes some attributes on the handle's ``FlatParameter``.
        This should be called during lazy initialization since it requires the
        parameter to be on the compute device if not offloading to CPU and we
        want to give users the chance to move the parameter appropriately after
        the FSDP constructor.

        For each tensor attribute on the ``FlatParameter``, see the unshard and
        reshard methods in this class for the allocation and free pattern.
        """
        flat_param = self.flat_param
        if flat_param.dtype != self._orig_param_dtype:
            if not self._low_prec_param_dtype_specified:
                self._fwd_bwd_param_dtype = flat_param.dtype
            if not self._low_prec_reduce_dtype_specified and (not self._low_prec_param_dtype_specified):
                self._reduce_dtype = flat_param.dtype
            self._orig_param_dtype = flat_param.dtype
        cpu_device = torch.device('cpu')
        if self._offload_params:
            _p_assert(flat_param.device == cpu_device, f'Expects the `FlatParameter` to be on CPU when parameter CPU offloading is enabled, not {flat_param.device}')
        else:
            self._check_on_compute_device(self.flat_param)
        flat_param._local_shard = flat_param.data
        if self._offload_params:
            flat_param._local_shard = flat_param._local_shard.pin_memory()
            flat_param._cpu_grad = torch.zeros_like(flat_param._local_shard, device=cpu_device).pin_memory()
        if self._uses_param_mixed_precision:
            flat_param._mp_shard = torch.empty_like(flat_param._local_shard, device=self.device, dtype=self._fwd_bwd_param_dtype)
            _free_storage(flat_param._mp_shard)
        if self.uses_sharded_strategy:
            unsharded_param_dtype = self._fwd_bwd_param_dtype if self._uses_param_mixed_precision else flat_param.dtype
            padded_unsharded_numel = flat_param.numel() * self.world_size
            flat_param._full_param_padded = torch.empty(padded_unsharded_numel, device=self.device, dtype=unsharded_param_dtype)
            flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
            _free_storage(flat_param._full_param_padded)
            if self._uses_param_mixed_precision:
                flat_param._full_prec_full_param_padded = torch.empty(padded_unsharded_numel, device=self.device, dtype=flat_param.dtype)
                _free_storage(flat_param._full_prec_full_param_padded)

    def pre_unshard(self) -> bool:
        """
        Returns: ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
        if self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS and self._skipped_use_sharded_views:
            self._use_sharded_views()
        ret = False
        if self._use_orig_params and (not self._skip_writeback_check):
            ret = self._writeback_orig_params()
        if self.uses_sharded_strategy and (not self._offload_params) and (not self.needs_unshard()):
            pass
        elif self._uses_param_mixed_precision and (not self._force_full_precision):
            self._use_low_precision_shard()
            ret = True
        elif self._offload_params and self.flat_param.device != self.device:
            self.flat_param_to(self.device, non_blocking=True)
            ret = True
        self._check_on_compute_device(self.flat_param)
        return ret

    def _use_low_precision_shard(self):
        """
        Allocates the low precision shard directly on the compute device and
        switches to using the low precision sharded flat parameter.
        """
        self._check_low_precision_shard()
        flat_param = self.flat_param
        _alloc_storage(flat_param._mp_shard, flat_param._local_shard.size())
        flat_param._mp_shard.copy_(flat_param._local_shard.to(self.device, non_blocking=True))
        flat_param.data = flat_param._mp_shard

    def unshard(self):
        """
        Runs the unshard logic. This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            unsharded_flat_param = self._get_padded_unsharded_flat_param() if self.uses_sharded_strategy else self.flat_param
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def needs_unshard(self) -> bool:
        """Returns if the handle's flat parameter needs to be unsharded."""
        if not self.uses_sharded_strategy:
            return False
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        already_unsharded = unsharded_flat_param._typed_storage()._size() == unsharded_flat_param.numel()
        return not already_unsharded

    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocates the *padded* unsharded flat parameter. The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_freed(unsharded_flat_param)
        _alloc_storage(unsharded_flat_param, flat_param._padded_unsharded_size)
        return unsharded_flat_param

    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """
        Returns a reference to the padded unsharded flat parameter depending on
        the calling context. This should only be called if using a sharded
        strategy.
        """
        self._check_sharded_strategy()
        flat_param = self.flat_param
        if self._force_full_precision and self._uses_param_mixed_precision:
            unsharded_flat_param = flat_param._full_prec_full_param_padded
            _p_assert(unsharded_flat_param.dtype != self._fwd_bwd_param_dtype, f'Expects full precision but got {self._fwd_bwd_param_dtype}')
            if flat_param._full_param_padded.untyped_storage().size() > 0:
                _free_storage(flat_param._full_param_padded)
        else:
            unsharded_flat_param = flat_param._full_param_padded
        return unsharded_flat_param

    def _all_gather_flat_param(self, padded_unsharded_flat_param: Tensor) -> Tensor:
        """
        All-gathers the handle's flat parameter to the destination
        ``padded_unsharded_flat_param``, and switches to using the all-gathered
        tensor.
        """
        _p_assert(hasattr(self, 'process_group') and hasattr(self, 'world_size'), 'Expects a process group and world size to have been set via `shard()`')
        sharded_flat_param = self.flat_param.data
        expected_numel = sharded_flat_param.numel() * self.world_size
        _p_assert(padded_unsharded_flat_param.numel() == expected_numel, f'Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}')
        if sharded_flat_param.is_cpu:
            tensor_list = list(torch.chunk(padded_unsharded_flat_param, dist.get_world_size(self.process_group)))
            work = dist.all_gather(tensor_list, sharded_flat_param, group=self.process_group)
        else:
            dist.all_gather_into_tensor(padded_unsharded_flat_param, sharded_flat_param, self.process_group)
        return padded_unsharded_flat_param

    def _use_unsharded_flat_param(self, padded_unsharded_flat_param: torch.Tensor) -> None:
        """
        Switches to using the *unpadded* unsharded flat parameter, which is a
        view into the *padded* unsharded flat parameter.
        """
        unsharded_size = self.flat_param._unpadded_unsharded_size
        self.flat_param.data = padded_unsharded_flat_param[:unsharded_size.numel()].view(unsharded_size)
        in_forward = self._training_state == HandleTrainingState.FORWARD
        in_pre_backward = self._training_state == HandleTrainingState.BACKWARD_PRE
        if self._use_orig_params:
            if self._skipped_use_sharded_views and in_pre_backward:
                return
            self._use_unsharded_views(as_params=not in_forward and (not in_pre_backward))
        elif in_forward:
            self._use_unsharded_views(as_params=False)

    def post_unshard(self):
        """
        Runs the post-unshard logic. This includes freeing the low precision
        shard if needed.
        """
        if self._uses_param_mixed_precision and self.uses_sharded_strategy:
            self._free_low_precision_sharded_param()
        self._check_on_compute_device(self.flat_param)

    def _free_low_precision_sharded_param(self):
        """Frees the low precision sharded flat parameter."""
        self._check_low_precision_shard()
        _no_dispatch_record_stream(self.flat_param._mp_shard, self._device_handle.current_stream())
        _free_storage(self.flat_param._mp_shard)

    @torch.no_grad()
    def unshard_grad(self):
        """
        Unshards the handle's ``FlatParameter`` 's gradient. If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
        if not self.uses_sharded_strategy:
            self._use_unsharded_grad_views()
            return
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        num_grad_none = torch.zeros(1, dtype=torch.int32, device=self.device)
        num_grad_none[0] = flat_param.grad is None
        dist.all_reduce(num_grad_none, group=self.process_group)
        if num_grad_none[0] == self.world_size:
            flat_param._saved_grad_shard = None
            self._use_unsharded_grad_views()
            return
        padded_unsharded_grad = torch.empty(flat_param._padded_unsharded_size, device=self.device)
        if flat_param.grad is None:
            if self._debug_level == dist.DebugLevel.INFO:
                warnings.warn(f"[Rank {self.rank}] Only some but not all ranks have a `None` `FlatParameter` gradient, so FSDP is using zeros to approximate those ranks' sharded gradients being `None`")
            flat_param._saved_grad_shard = None
            sharded_grad = torch.zeros(flat_param._sharded_size, device=self.device)
        else:
            self._check_sharded(flat_param.grad)
            flat_param._saved_grad_shard = flat_param.grad
            sharded_grad = flat_param._saved_grad_shard
        dist.all_gather_into_tensor(padded_unsharded_grad, sharded_grad, self.process_group)
        unsharded_size = self.flat_param._unpadded_unsharded_size
        flat_param.grad = padded_unsharded_grad[:unsharded_size.numel()].view(unsharded_size)
        self._use_unsharded_grad_views()

    def reshard_grad(self):
        if self._use_orig_params:
            self._use_sharded_grad_views()
        if not self.uses_sharded_strategy:
            return
        self.flat_param.grad = self.flat_param._saved_grad_shard
        delattr(self.flat_param, '_saved_grad_shard')

    def prepare_gradient_for_backward(self):
        """
        Prepares the gradient for the backward computation by saving and
        clearing any existing sharded gradient in ``.grad`` to enable computing
        a new unsharded gradient.
        """
        _p_assert(self._training_state in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.IDLE), 'Expects to be in `BACKWARD_PRE` or `IDLE` (if prefetching)')
        flat_param = self.flat_param
        if flat_param.grad is not None and (flat_param.grad.size() != flat_param._unpadded_unsharded_size or flat_param.grad.device != flat_param.device):
            self._check_on_compute_device(self.flat_param)
            grad_offloaded = flat_param.grad.device != self.device
            _p_assert(not grad_offloaded or self._offload_params, f'Expects the sharded gradient to be on {self.device} but got {flat_param.grad.device}')
            prev_iter_synced_gradients = flat_param.grad.size() == flat_param._local_shard.size()
            if prev_iter_synced_gradients:
                if not grad_offloaded:
                    flat_param._saved_grad_shard = flat_param.grad.data
                    sharded_grad = flat_param._saved_grad_shard
                else:
                    _p_assert(hasattr(flat_param, '_cpu_grad'), '`_cpu_grad` should be defined if the gradient is on CPU')
                    sharded_grad = flat_param._cpu_grad
                local_shard_dtype = flat_param._local_shard.dtype
                if self._keep_low_precision_grads and sharded_grad.dtype != local_shard_dtype:
                    sharded_grad.data = sharded_grad.to(local_shard_dtype)
            else:
                padded_unsharded_size = flat_param._padded_unsharded_size
                _p_assert(flat_param.grad.size() == padded_unsharded_size, f'Expects `.grad` to be the unsharded gradient in `no_sync()` with size {padded_unsharded_size} but got size {flat_param.grad.size()}')
            flat_param.grad = None

    def prepare_gradient_for_optim(self):
        """
        Prepares the gradient for optimizer computation by moving the sharded
        gradient to the ``.grad`` attribute.
        """

        def cast_grad_to_param_dtype_if_needed(flat_param):
            if not self._force_full_precision and self._keep_low_precision_grads:
                _p_assert(flat_param.grad is not None, 'Unexpected None grad!')
                if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                    flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                    if self._use_orig_params:
                        self._use_sharded_grad_views()
        flat_param = self.flat_param
        if hasattr(flat_param, '_cpu_grad'):
            self._check_sharded(flat_param)
            self._check_on_cpu(flat_param)
            flat_param.grad = flat_param._cpu_grad
            cast_grad_to_param_dtype_if_needed(flat_param)
        elif hasattr(flat_param, '_saved_grad_shard'):
            self._check_sharded(flat_param)
            self._check_on_compute_device(flat_param)
            if flat_param._saved_grad_shard is not None:
                self._check_on_compute_device(flat_param._saved_grad_shard)
            if flat_param._post_backward_called:
                flat_param.grad = flat_param._saved_grad_shard
                if flat_param.grad is not None:
                    cast_grad_to_param_dtype_if_needed(flat_param)
        else:
            _p_assert(not self.uses_sharded_strategy or not flat_param._post_backward_called, 'All sharded parameters that received a gradient in the post-backward should use `_saved_grad_shard`')
        if hasattr(flat_param, '_saved_grad_shard'):
            delattr(flat_param, '_saved_grad_shard')

    @contextlib.contextmanager
    def to_cpu(self):
        """
        Moves the unpadded unsharded flat parameter to CPU while in the context
        and moves it back to the previous device upon exit. For now, this
        assumes the ``FlatParameter`` is the unpadded unsharded flat parameter
        since (1) there is no reason to include the padding in the copy and (2)
        there is no use case for the sharded flat parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flat parameter on the compute device, and the handle uses a sharded
        strategy.
        Postcondition: Same as the precondition.
        """
        self._check_sharded_strategy()
        _p_assert(self.flat_param.size() == self.flat_param._unpadded_unsharded_size, f'Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}')
        self._check_on_compute_device(self.flat_param)
        unpadded_storage_ptr = self.flat_param._typed_storage()._data_ptr()
        padded_storage_ptr = self._get_padded_unsharded_flat_param()._typed_storage()._data_ptr()
        _p_assert(unpadded_storage_ptr == padded_storage_ptr, 'Expects the unpadded parameter to be a view into the padded parameter')
        self.flat_param_to(torch.device('cpu'))
        self._free_unsharded_flat_param()
        try:
            yield
        finally:
            _p_assert(self.flat_param.size() == self.flat_param._unpadded_unsharded_size, f'Expects size {self.flat_param._unpadded_unsharded_size} but got {self.flat_param.size()}')
            padded_unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
            padded_unsharded_flat_param[:self.flat_param.numel()].copy_(self.flat_param)
            self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def reshard(self, free_unsharded_flat_param: bool):
        """
        Runs the reshard logic. This includes freeing the unsharded flat
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flat parameter. Note that this also implicitly offloads
        the sharded flat parameter (if CPU offload is enabled) by pointing
        it to the ``_local_shard`` attribute which resides on CPU.
        """
        self._use_sharded_flat_param()
        if free_unsharded_flat_param:
            self._free_unsharded_flat_param()

    def post_reshard(self):
        """
        Runs the post-reshard logic. This includes freeing any memory that
        can now be freed given that the ``FlatParameter`` points to the full
        precision sharded flat parameter.

        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flat parameter.
        """
        if self._uses_param_mixed_precision and (not self.uses_sharded_strategy) and (not self._force_full_precision):
            self._free_low_precision_sharded_param()

    def _free_unsharded_flat_param(self):
        """
        Frees the padded unsharded flat parameter. The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        self._check_sharded_strategy()
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_allocated(unsharded_flat_param)
        self._check_on_compute_device(unsharded_flat_param)
        _no_dispatch_record_stream(unsharded_flat_param, self._device_handle.current_stream())
        _free_storage(unsharded_flat_param)

    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flat parameter."""
        flat_param = self.flat_param
        if self._use_orig_params:
            in_forward = self._training_state == HandleTrainingState.FORWARD
            skip_use_sharded_views = torch.is_grad_enabled() and in_forward and (self._sharding_strategy in NO_RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES)
            if skip_use_sharded_views:
                unsharded_flat_param = flat_param.data
        if self._offload_params:
            device = flat_param._local_shard.device
            _p_assert(device == torch.device('cpu'), f'Expects the local shard to be on CPU but got {device}')
        flat_param.data = flat_param._local_shard
        if self._use_orig_params:
            if skip_use_sharded_views:
                self._unsharded_flat_param_for_skipped_views = unsharded_flat_param
            else:
                self._use_sharded_views()
            if in_forward and (not self._skipped_use_sharded_views):
                accumulated_grad_in_no_sync = flat_param.grad is not None and self.uses_sharded_strategy and (flat_param.grad.shape == flat_param._unpadded_unsharded_size)
                if accumulated_grad_in_no_sync:
                    self._use_unsharded_grad_views()
                else:
                    self._use_sharded_grad_views()

    @no_type_check
    def _get_unflat_views_unaligned(self, tensor: Optional[torch.Tensor]=None) -> Iterator[Tensor]:
        """
        Returns unflattened ``Tensor`` views into ``tensor`` if it is not
        ``None`` or ``flat_param`` otherwise, where the unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        views = (_ext_post_unflatten_transform(subtensor.view(shape), param_extension) for subtensor, shape, param_extension in zip(torch.split(tensor, flat_param._numels, dim=0), flat_param._shapes, flat_param._param_extensions))
        return views

    @no_type_check
    def _get_unflat_views_aligned(self, tensor: Optional[Tensor]=None) -> List[Tensor]:
        """
        This has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        splits: List[Tensor] = torch.split(tensor, flat_param._numels_with_padding, dim=0)
        idx = 0
        views: List[Tensor] = []
        for split, is_padding in zip(splits, flat_param._is_padding_mask):
            if is_padding:
                continue
            views.append(_ext_post_unflatten_transform(split.view(flat_param._shapes[idx]), flat_param._param_extensions[idx]))
            idx += 1
        return views

    @no_type_check
    def _use_unsharded_views(self, as_params: bool) -> None:
        """
        Unflattens the unsharded flat parameter by setting the original
        parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.
        """
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        views = self._get_unflat_views()
        from torch.distributed._tensor import DTensor
        for i, (view, (param_name, module, _)) in enumerate(zip(views, flat_param._param_infos)):
            if self._use_orig_params and as_params:
                if type(view) is DTensor:
                    self._setattr_param(module, param_name, nn.Parameter(view))
                    continue
                param = self.flat_param._params[i]
                self._setattr_param(module, param_name, param)
                param.data = view
            elif as_params:
                self._setattr_param(module, param_name, nn.Parameter(view))
            else:
                param_var: Tensor = view
                if self._use_orig_params:
                    if self._training_state == HandleTrainingState.FORWARD:
                        self.flat_param._tensors[i] = view
                    elif self._training_state == HandleTrainingState.BACKWARD_PRE:
                        tensor = self.flat_param._tensors[i]
                        tensor.data = view
                        param_var = tensor
                self._setattr_tensor(module, param_name, param_var)
                if self._use_orig_params and self._training_state == HandleTrainingState.FORWARD:
                    module._parameters[param_name] = param_var
        for i, (param_name, module, _, prim_param_name, prim_module, _) in enumerate(self.flat_param._shared_param_infos):
            prim_param: Union[Tensor, nn.Parameter] = getattr(prim_module, prim_param_name)
            _p_assert(not as_params or isinstance(prim_param, nn.Parameter), f'as_params={as_params} type(prim_param)={type(prim_param)}')
            if self._use_orig_params and as_params:
                shared_param = self.flat_param._shared_params[i]
                self._setattr_param(module, param_name, shared_param)
                shared_param.data = prim_param
            elif as_params:
                self._setattr_param(module, param_name, prim_param)
            else:
                self._setattr_tensor(module, param_name, prim_param)
                if self._use_orig_params and self._training_state == HandleTrainingState.FORWARD:
                    module._parameters[param_name] = prim_param

    @no_type_check
    def _use_unsharded_grad_views(self) -> None:
        """
        Unflattens the unsharded flat parameter's gradient by setting the
        original parameter variables' gradients to be views into it.
        """
        if self.flat_param.grad is None:
            for param in chain(self.flat_param._params, self.flat_param._shared_params):
                param.grad = None
            return
        self._check_unsharded(self.flat_param.grad)
        views = self._get_unflat_views(self.flat_param.grad)
        for i, (view, (param_name, module, _)) in enumerate(zip(views, self.flat_param._param_infos)):
            _p_assert(hasattr(module, param_name), f'{self.flat_param._fqns[i]} is missing')
            param = getattr(module, param_name)
            if param.shape != view.shape or param.dtype != view.dtype or param.device != view.device:
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = view
            else:
                param.grad = view
        for i, (param_name, module, module_name, prim_param_name, prim_module, _) in enumerate(self.flat_param._shared_param_infos):
            _p_assert(hasattr(module, param_name), f'{(module_name + '.' + param_name if module_name else param_name)} is missing')
            param = getattr(module, param_name)
            prim_param = getattr(prim_module, prim_param_name)
            if param.shape != prim_param.grad.shape or param.dtype != prim_param.grad.dtype or param.device != prim_param.grad.device:
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.data = prim_param.grad
            else:
                param.grad = prim_param.grad

    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Assumes the flat parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flat parameter, and after the context, restores the original parameters
        as ``Tensor`` views into the flat parameter.
        """
        self._use_unsharded_views(as_params=True)
        try:
            yield
        finally:
            self._use_unsharded_views(as_params=False)

    @no_type_check
    @torch.no_grad()
    def _use_sharded_views(self) -> None:
        """
        Sets the original parameter variables' data to be flattened views into
        the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
        self._unsharded_flat_param_for_skipped_views = None
        if not self.uses_sharded_strategy:
            self._use_unsharded_views(as_params=True)
            return
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        size_0_empty_tensor = torch.empty(0, dtype=self.flat_param.dtype, device=self.flat_param.device, requires_grad=False)
        for param, shard_param_info, (param_name, module, _) in zip(flat_param._params, flat_param._shard_param_infos, flat_param._param_infos):
            self._setattr_param(module, param_name, param)
            if not shard_param_info.in_shard:
                param.data = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                param.data = flat_param[offset:offset + numel_in_shard]
        assert self.flat_param._shared_params is not None
        for i, (param, (param_name, module, _, prim_param_name, prim_module, _)) in enumerate(zip(self.flat_param._shared_params, self.flat_param._shared_param_infos)):
            self._setattr_param(module, param_name, param)
            prim_param = getattr(prim_module, prim_param_name)
            param.data = prim_param
        if self._training_state == HandleTrainingState.BACKWARD_POST:
            for i in range(len(self.flat_param._tensors)):
                self.flat_param._tensors[i] = None

    @no_type_check
    @torch.no_grad()
    def _use_sharded_grad_views(self) -> None:
        """
        Sets the original parameter variables' gradients to be flattened
        views into the sharded flat parameter's gradient. This is a no-op if
        there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
        flat_param = self.flat_param
        self._check_sharded(flat_param)
        grad = self.sharded_grad
        if grad is None:
            for param in chain(flat_param._params, flat_param._shared_params):
                param.grad = None
            return
        self._check_sharded(grad)
        for param, shard_param_info, is_grad_none in zip(flat_param._params, flat_param._shard_param_infos, flat_param._is_grad_none_mask):
            if not shard_param_info.in_shard:
                param.grad = None
            else:
                numel_in_shard = shard_param_info.numel_in_shard
                if param.requires_grad and (not is_grad_none):
                    offset = shard_param_info.offset_in_shard
                    if self._keep_low_precision_grads or param.dtype != grad.dtype:
                        if param.grad is None:
                            param.grad = torch.empty_like(param)
                        param.grad.data = grad[offset:offset + numel_in_shard].reshape(param.shape)
                    else:
                        param.grad = grad[offset:offset + numel_in_shard].reshape(param.shape)
                else:
                    param.grad = None
        assert flat_param._shared_params is not None
        for i, (param, (_, _, _, prim_param_name, prim_module, _)) in enumerate(zip(flat_param._shared_params, flat_param._shared_param_infos)):
            in_sharded_flat_param = hasattr(prim_module, prim_param_name)
            if in_sharded_flat_param and param.requires_grad:
                prim_param = getattr(prim_module, prim_param_name)
                param.grad = prim_param.grad
            else:
                param.grad = None

    @no_type_check
    @torch.no_grad()
    def _writeback_orig_params(self) -> bool:
        """
        Iterates over the original parameters and writes back any parameters
        that changed storages (due to a non-inplace operator) to the handle's
        ``FlatParameter``. This method preserves the ``FlatParameter` 's
        device even if an original parameter's device changes.

        Raises:
            RuntimeError: If an original parameter or gradient changes storages
            but no longer has the expected flattened shape.
        Returns: ``True`` if some writeback happened, and ``False`` otherwise.
        """
        if self.uses_sharded_strategy and (not self.is_sharded(self.flat_param)) and (not self._skipped_use_sharded_views):
            return False
        flat_param = self.flat_param
        wroteback = False
        if self._skipped_use_sharded_views and self.uses_sharded_strategy:
            flat_param_data_ptr = self._unsharded_flat_param_for_skipped_views.untyped_storage().data_ptr()
            _p_assert(flat_param_data_ptr > 0, 'If skipped using sharded views, the unsharded flat parameter should be allocated')
        else:
            flat_param_data_ptr = flat_param.untyped_storage().data_ptr()
        flat_param_grad = flat_param.grad if self.uses_sharded_strategy or not self._offload_params else flat_param._cpu_grad
        flat_param_grad_data_ptr = None if flat_param_grad is None else flat_param_grad.untyped_storage().data_ptr()
        for i, (param, (in_shard, offset_in_shard, numel_in_shard, _, _), (param_name, module, _)) in enumerate(zip(flat_param._params, flat_param._shard_param_infos, flat_param._param_infos)):
            if not in_shard:
                continue
            if not hasattr(module, param_name):
                continue
            if self._skipped_use_sharded_views:
                param = flat_param._tensors[i]
                _p_assert(param is not None, f'Expects to have saved tensor for {flat_param._fqns[i]}')
            param_changed = getattr(module, param_name) is not param
            needs_param_writeback = param_changed or not _same_storage_as_data_ptr(param, flat_param_data_ptr)
            if self._skipped_use_sharded_views and (param_changed or needs_param_writeback):
                raise AssertionError(f'FSDP does not support changing the parameters between forward and backward for {self._sharding_strategy}')
            if param_changed:
                param = getattr(module, param_name)
                flat_param._params[i] = param
            if needs_param_writeback:
                expected_shape = torch.Size([numel_in_shard])
                self._writeback_tensor(param, flat_param, i, expected_shape, offset_in_shard, True)
                wroteback = True
            if self._skipped_use_sharded_views:
                continue
            if param.grad is None and flat_param.grad is not None:
                expected_shape = torch.Size([numel_in_shard])
                self._writeback_tensor(None, flat_param.grad, i, expected_shape, offset_in_shard, False)
            elif param.grad is not None:
                if not self.uses_sharded_strategy and self._offload_params:
                    continue
                needs_grad_writeback = flat_param_grad is None or not _same_storage_as_data_ptr(param.grad, flat_param_grad_data_ptr)
                if needs_grad_writeback:
                    if flat_param_grad is None:
                        flat_param_grad = torch.zeros_like(flat_param)
                    expected_shape = torch.Size([numel_in_shard])
                    self._writeback_tensor(param.grad, flat_param_grad, i, expected_shape, offset_in_shard, False)
                    flat_param.grad = flat_param_grad
                    flat_param_grad = flat_param.grad
                    flat_param_grad_data_ptr = flat_param_grad.untyped_storage().data_ptr()
        for i, (param_name, module, _, prim_param_name, prim_module, _) in enumerate(flat_param._shared_param_infos):
            if getattr(module, param_name) is not getattr(prim_module, prim_param_name):
                raise NotImplementedError('Changing shared parameters is not supported yet')
        return wroteback

    def _writeback_tensor(self, src_tensor: Optional[Tensor], dst_tensor: Tensor, tensor_index: int, expected_shape: torch.Size, offset: int, is_param: bool) -> None:
        """
        Writes back ``src_tensor`` to ``dst_tensor`` at offset ``offset``,
        where ``src_tensor`` should have shape ``expected_shape``. ``is_param``
        indicates if the tensor is the parameter (if ``True``) or gradient (if
        ``False``). If ``src_tensor`` is ``None``, then the effect is zeroing
        instead of copying. ``tensor_index`` gives the index of ``src_tensor``
        in the metadata structures.

        Raises:
            RuntimeError: If the ``src_tensor`` does not have the expected
            shape.
        """
        _p_assert(len(expected_shape) == 1, f'Expects a 1D expected shape but got {expected_shape}')
        if self._debug_level == dist.DebugLevel.INFO:
            rank = self.rank if hasattr(self, 'rank') else dist.get_rank()
            src_shape = src_tensor.shape if src_tensor is not None else None
            src_device = src_tensor.device if src_tensor is not None else None
            warnings.warn(f'[Rank {rank}] {('Parameter' if is_param else 'Gradient')} needs writeback in {self._training_state}\nexpected shape={expected_shape} shape={src_shape} expected device={dst_tensor.device} device={src_device}')
        if src_tensor is not None and src_tensor.shape != expected_shape:
            raise RuntimeError(f'Cannot writeback when the {('parameter' if is_param else 'gradient')} shape changes\nExpects {expected_shape} but got {src_tensor.shape}')
        if src_tensor is not None:
            dst_tensor[offset:offset + expected_shape.numel()].copy_(src_tensor)
        else:
            dst_tensor[offset:offset + expected_shape.numel()].zero_()
            assert self.flat_param._is_grad_none_mask is not None
            self.flat_param._is_grad_none_mask[tensor_index] = True

    def _reset_flat_param_grad_info_if_needed(self):
        """
        When ``use_orig_params=True``:
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
        if not self._use_orig_params:
            return
        flat_param = self.flat_param
        assert flat_param._params is not None
        all_grad_none = True
        requires_grad = False
        for param in flat_param._params:
            all_grad_none &= param.grad is None
            requires_grad |= param.requires_grad
        if all_grad_none:
            flat_param.grad = None
        flat_param.requires_grad = requires_grad

    def _deregister_orig_params(self):
        for param_info in self.flat_param._param_infos:
            param_name, module, _ = param_info
            if hasattr(module, param_name):
                delattr(module, param_name)
        for param_name, module, _, _, _, _ in self.flat_param._shared_param_infos:
            if hasattr(module, param_name):
                delattr(module, param_name)

    def flat_param_to(self, *args, **kwargs):
        """Wraps an in-place call to ``.to()`` for ``self.flat_param``."""
        self.flat_param.data = self.flat_param.to(*args, **kwargs)
        if self._use_orig_params:
            if self.is_sharded(self.flat_param):
                self._use_sharded_views()
            else:
                self._use_unsharded_views(as_params=True)

    def _get_modules(self) -> Set[nn.Module]:
        """
        Returns a :class:`set` of the modules whose parameters are included
        in this handle's flat parameter.
        """
        return {pi.module for pi in self.flat_param._param_infos}.union({spi.module for spi in self.flat_param._shared_param_infos})

    def is_sharded(self, tensor: Tensor) -> bool:
        """
        Returns if ``tensor`` is *currently* sharded. For ``NO_SHARD``, we
        choose to have this always return ``False`` for clarity.
        """
        if not hasattr(self.flat_param, '_sharded_size') or not self.uses_sharded_strategy:
            return False
        sharded_size = self.flat_param._sharded_size
        return tensor.size() == sharded_size

    def param_module_names(self) -> Iterator[Tuple[str, str]]:
        shared_param_infos = [ParamInfo(param_name, module, module_name) for param_name, module, module_name, _, _, _ in self.flat_param._shared_param_infos]
        for param_info in chain(self.flat_param._param_infos, shared_param_infos):
            param_name, _, module_name = param_info
            yield (param_name, module_name)

    def shared_param_module_names(self) -> Iterator[Tuple[str, str]]:
        for param_name, _, module_name in [ParamInfo(param_name, module, module_name) for param_name, module, module_name, _, _, _ in self.flat_param._shared_param_infos]:
            yield (param_name, module_name)

    @property
    def _fqns_in_shard(self) -> List[str]:
        """Returns the FQNs of the parameters present in this rank's shard."""
        fqns_in_shard: List[str] = []
        for fqn, shard_param_info in zip(self.flat_param._fqns, self.flat_param._shard_param_infos):
            if shard_param_info.in_shard:
                fqns_in_shard.append(fqn)
        return fqns_in_shard

    @property
    def sharded_grad(self) -> Optional[Tensor]:
        """Returns the handle's sharded gradient."""
        flat_param = self.flat_param
        grad: Optional[Tensor]
        if hasattr(flat_param, '_cpu_grad'):
            grad = flat_param._cpu_grad
        elif hasattr(flat_param, '_saved_grad_shard'):
            grad = flat_param._saved_grad_shard
        else:
            _p_assert(flat_param.grad is None or not self.uses_sharded_strategy or self._training_state in (HandleTrainingState.FORWARD, HandleTrainingState.IDLE), 'Sharded strategies should use `_cpu_grad` or `_saved_grad_shard` unless in IDLE or FORWARD')
            grad = flat_param.grad
        return grad

    def _reset_is_grad_none(self) -> None:
        """
        Resets ``_is_grad_none_mask`` as needed. This method should only be
        called in the post-backward after gradient computation, in which case
        if a parameter requires gradient, then it will surely receive a
        gradient and we may reset its mask entry to ``False``.
        """
        if not self._use_orig_params:
            return
        _p_assert(self._training_state == HandleTrainingState.BACKWARD_POST, 'Expects to only be called in the post-backward after gradient computation')
        flat_param = self.flat_param
        assert flat_param._params is not None
        for i, param in enumerate(flat_param._params):
            if param.requires_grad:
                assert flat_param._is_grad_none_mask is not None
                flat_param._is_grad_none_mask[i] = False

    def _check_sharded_strategy(self):
        _p_assert(self.uses_sharded_strategy, 'Expects sharded strategy')

    def _check_on_compute_device(self, tensor: Tensor):
        _p_assert(tensor.device == self.device, f'Expects tensor to be on the compute device {self.device}')

    def _check_on_cpu(self, tensor: Tensor):
        _p_assert(tensor.device == torch.device('cpu'), f'Expects tensor to be on CPU but got {tensor.device}')

    @staticmethod
    def _check_storage_freed(tensor: Tensor):
        storage_size: int = tensor._typed_storage()._size()
        _p_assert(storage_size == 0, f'Expects storage to be freed but got storage with size {storage_size}')

    @staticmethod
    def _check_storage_allocated(tensor: Tensor):
        storage_size: int = tensor._typed_storage()._size()
        _p_assert(storage_size > 0, 'Expects storage to be allocated')

    def _check_low_precision_shard(self):
        _p_assert(self._uses_param_mixed_precision, 'Not using low precision for parameters')
        _p_assert(getattr(self.flat_param, '_mp_shard', None) is not None, 'Expects `_mp_shard` to exist')
        device = self.flat_param._mp_shard.device
        _p_assert(device == self.device, f'Expects the low precision shard to be on {self.device} but got {device}')

    def _check_unsharded(self, tensor: Tensor):
        msg_prefix = 'Expects tensor to be unsharded '
        _p_assert(tensor is not None, msg_prefix + 'but got `None`')
        unsharded_size = self.flat_param._unpadded_unsharded_size
        _p_assert(tensor.size() == unsharded_size, msg_prefix + f'with size {unsharded_size} but got {tensor.size()}')

    def _check_sharded(self, tensor: Tensor):
        msg_prefix = 'Expects tensor to be sharded '
        _p_assert(tensor is not None, msg_prefix + 'but got `None`')
        sharded_size = self.flat_param._sharded_size
        _p_assert(tensor.size() == sharded_size, msg_prefix + f'with size {sharded_size} but got {tensor.size()}')

    @property
    def uses_sharded_strategy(self) -> bool:
        return self._sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def _uses_param_mixed_precision(self) -> bool:
        return self._fwd_bwd_param_dtype != self._orig_param_dtype

    @property
    def _uses_reduce_mixed_precision(self) -> bool:
        return self._reduce_dtype != self._orig_param_dtype

    @property
    def _force_full_precision(self) -> bool:
        return (self._uses_param_mixed_precision or self._uses_reduce_mixed_precision) and (self._training_state == HandleTrainingState.SUMMON_FULL_PARAMS or (not self._fully_sharded_module.training and self._use_full_prec_in_eval))

    @property
    def _skipped_use_sharded_views(self) -> bool:
        """
        This property is used for sharding strategies that do not free after
        forward with ``use_orig_params=True``. This returns if this handle is
        currently in a state where it has skipped using sharded views, in which
        case it can restore view invariants via ``_use_sharded_views()``.
        """
        return self._unsharded_flat_param_for_skipped_views is not None