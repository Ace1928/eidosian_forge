from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class MixedPrecision:
    """
    This configures FSDP-native mixed precision training.

    Attributes:
        param_dtype (Optional[torch.dtype]): This specifies the dtype for model
            parameters during forward and backward and thus the dtype for
            forward and backward computation. Outside forward and backward, the
            *sharded* parameters are kept in full precision (e.g. for the
            optimizer step), and for model checkpointing, the parameters are
            always saved in full precision. (Default: ``None``)
        reduce_dtype (Optional[torch.dtype]): This specifies the dtype for
            gradient reduction (i.e. reduce-scatter or all-reduce). If this is
            ``None`` but ``param_dtype`` is not ``None``, then this takes on
            the ``param_dtype`` value, still running gradient reduction in low
            precision. This is permitted to differ from ``param_dtype``, e.g.
            to force gradient reduction to run in full precision. (Default:
            ``None``)
        buffer_dtype (Optional[torch.dtype]): This specifies the dtype for
            buffers. FSDP does not shard buffers. Rather, FSDP casts them to
            ``buffer_dtype`` in the first forward pass and keeps them in that
            dtype thereafter. For model checkpointing, the buffers are saved
            in full precision except for ``LOCAL_STATE_DICT``. (Default:
            ``None``)
        keep_low_precision_grads (bool): If ``False``, then FSDP upcasts
            gradients to full precision after the backward pass in preparation
            for the optimizer step. If ``True``, then FSDP keeps the gradients
            in the dtype used for gradient reduction, which can save memory if
            using a custom optimizer that supports running in low precision.
            (Default: ``False``)
        cast_forward_inputs (bool): If ``True``, then this FSDP module casts
            its forward args and kwargs to ``param_dtype``. This is to ensure
            that parameter and input dtypes match for forward computation, as
            required by many ops. This may need to be set to ``True`` when only
            applying mixed precision to some but not all FSDP modules, in which
            case a mixed-precision FSDP submodule needs to recast its inputs.
            (Default: ``False``)
        cast_root_forward_inputs (bool): If ``True``, then the root FSDP module
            casts its forward args and kwargs to ``param_dtype``, overriding
            the value of ``cast_forward_inputs``. For non-root FSDP modules,
            this does not do anything. (Default: ``True``)
        _module_classes_to_ignore: (Sequence[Type[nn.Module]]): This specifies
            module classes to ignore for mixed precision when using an
            ``auto_wrap_policy``: Modules of these classes will have FSDP
            applied to them separately with mixed precision disabled (meaning
            that the final FSDP construction would deviate from the specified
            policy). If ``auto_wrap_policy`` is not specified, then this does
            not do anything. This API is experimental and subject to change.
            (Default: ``(_BatchNorm,)``)

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: In ``summon_full_params``, parameters are forced to full
        precision, but buffers are not.

    .. note:: Layer norm and batch norm accumulate in ``float32`` even when
        their inputs are in a low precision like ``float16`` or ``bfloat16``.
        Disabling FSDP's mixed precision for those norm modules only means that
        the affine parameters are kept in ``float32``. However, this incurs
        separate all-gathers and reduce-scatters for those norm modules, which
        may be inefficient, so if the workload permits, the user should prefer
        to still apply mixed precision to those modules.

    .. note:: By default, if the user passes a model with any ``_BatchNorm``
        modules and specifies an ``auto_wrap_policy``, then the batch norm
        modules will have FSDP applied to them separately with mixed precision
        disabled. See the ``_module_classes_to_ignore`` argument.

    .. note:: ``MixedPrecision`` has ``cast_root_forward_inputs=True`` and
        ``cast_forward_inputs=False`` by default. For the root FSDP instance,
        its ``cast_root_forward_inputs`` takes precedence over its
        ``cast_forward_inputs``. For non-root FSDP instances, their
        ``cast_root_forward_inputs`` values are ignored. The default setting is
        sufficient for the typical case where each FSDP instance has the same
        ``MixedPrecision`` configuration and only needs to cast inputs to the
        ``param_dtype`` at the beginning of the model's forward pass.

    .. note:: For nested FSDP instances with different ``MixedPrecision``
        configurations, we recommend setting individual ``cast_forward_inputs``
        values to configure casting inputs or not before each instance's
        forward. In such a case, since the casts happen before each FSDP
        instance's forward, a parent FSDP instance should have its non-FSDP
        submodules run before its FSDP submodules to avoid the activation dtype
        being changed due to a different ``MixedPrecision`` configuration.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
            >>> model[1] = FSDP(
            >>>     model[1],
            >>>     mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True),
            >>> )
            >>> model = FSDP(
            >>>     model,
            >>>     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True),
            >>> )

        The above shows a working example. On the other hand, if ``model[1]``
        were replaced with ``model[0]``, meaning that the submodule using
        different ``MixedPrecision`` ran its forward first, then ``model[1]``
        would incorrectly see ``float16`` activations instead of ``bfloat16``
        ones.

    """
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    buffer_dtype: Optional[torch.dtype] = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True
    _module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = (_BatchNorm,)