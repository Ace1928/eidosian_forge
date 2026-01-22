from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
def _spatial_distortion_index_update(preds: Tensor, ms: Tensor, pan: Tensor, pan_lr: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Update and returns variables required to compute Spatial Distortion Index.

    Args:
        preds: High resolution multispectral image.
        ms: Low resolution multispectral image.
        pan: High resolution panchromatic image.
        pan_lr: Low resolution panchromatic image.

    Return:
        A tuple of Tensors containing ``preds``, ``ms``, ``pan`` and ``pan_lr``.

    Raises:
        TypeError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same data type.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have ``BxCxHxW shape``.
        ValueError:
            If ``preds``, ``ms``, ``pan`` and ``pan_lr`` don't have the same batch and channel sizes.
        ValueError:
            If ``preds`` and ``pan`` don't have the same dimension.
        ValueError:
            If ``ms`` and ``pan_lr`` don't have the same dimension.
        ValueError:
            If ``preds`` and ``pan`` don't have dimension which is multiple of that of ``ms``.

    """
    if len(preds.shape) != 4:
        raise ValueError(f'Expected `preds` to have BxCxHxW shape. Got preds: {preds.shape}.')
    if preds.dtype != ms.dtype:
        raise TypeError(f'Expected `preds` and `ms` to have the same data type. Got preds: {preds.dtype} and ms: {ms.dtype}.')
    if preds.dtype != pan.dtype:
        raise TypeError(f'Expected `preds` and `pan` to have the same data type. Got preds: {preds.dtype} and pan: {pan.dtype}.')
    if pan_lr is not None and preds.dtype != pan_lr.dtype:
        raise TypeError(f'Expected `preds` and `pan_lr` to have the same data type. Got preds: {preds.dtype} and pan_lr: {pan_lr.dtype}.')
    if len(ms.shape) != 4:
        raise ValueError(f'Expected `ms` to have BxCxHxW shape. Got ms: {ms.shape}.')
    if len(pan.shape) != 4:
        raise ValueError(f'Expected `pan` to have BxCxHxW shape. Got pan: {pan.shape}.')
    if pan_lr is not None and len(pan_lr.shape) != 4:
        raise ValueError(f'Expected `pan_lr` to have BxCxHxW shape. Got pan_lr: {pan_lr.shape}.')
    if preds.shape[:2] != ms.shape[:2]:
        raise ValueError(f'Expected `preds` and `ms` to have the same batch and channel sizes. Got preds: {preds.shape} and ms: {ms.shape}.')
    if preds.shape[:2] != pan.shape[:2]:
        raise ValueError(f'Expected `preds` and `pan` to have the same batch and channel sizes. Got preds: {preds.shape} and pan: {pan.shape}.')
    if pan_lr is not None and preds.shape[:2] != pan_lr.shape[:2]:
        raise ValueError(f'Expected `preds` and `pan_lr` to have the same batch and channel sizes. Got preds: {preds.shape} and pan_lr: {pan_lr.shape}.')
    preds_h, preds_w = preds.shape[-2:]
    ms_h, ms_w = ms.shape[-2:]
    pan_h, pan_w = pan.shape[-2:]
    if preds_h != pan_h:
        raise ValueError(f'Expected `preds` and `pan` to have the same height. Got preds: {preds_h} and pan: {pan_h}')
    if preds_w != pan_w:
        raise ValueError(f'Expected `preds` and `pan` to have the same width. Got preds: {preds_w} and pan: {pan_w}')
    if preds_h % ms_h != 0:
        raise ValueError(f'Expected height of `preds` to be multiple of height of `ms`. Got preds: {preds_h} and ms: {ms_h}.')
    if preds_w % ms_w != 0:
        raise ValueError(f'Expected width of `preds` to be multiple of width of `ms`. Got preds: {preds_w} and ms: {ms_w}.')
    if pan_h % ms_h != 0:
        raise ValueError(f'Expected height of `pan` to be multiple of height of `ms`. Got preds: {pan_h} and ms: {ms_h}.')
    if pan_w % ms_w != 0:
        raise ValueError(f'Expected width of `pan` to be multiple of width of `ms`. Got preds: {pan_w} and ms: {ms_w}.')
    if pan_lr is not None:
        pan_lr_h, pan_lr_w = pan_lr.shape[-2:]
        if pan_lr_h != ms_h:
            raise ValueError(f'Expected `ms` and `pan_lr` to have the same height. Got ms: {ms_h} and pan_lr: {pan_lr_h}.')
        if pan_lr_w != ms_w:
            raise ValueError(f'Expected `ms` and `pan_lr` to have the same width. Got ms: {ms_w} and pan_lr: {pan_lr_w}.')
    return (preds, ms, pan, pan_lr)