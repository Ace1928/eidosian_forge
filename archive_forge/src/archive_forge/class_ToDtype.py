import warnings
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from ._utils import _parse_labels_getter, _setup_number_or_seq, _setup_size, get_bounding_boxes, has_any, is_pure_tensor
class ToDtype(Transform):
    """[BETA] Converts the input to a specific dtype, optionally scaling the values for images or videos.

    .. v2betastatus:: ToDtype transform

    .. note::
        ``ToDtype(dtype, scale=True)`` is the recommended replacement for ``ConvertImageDtype(dtype)``.

    Args:
        dtype (``torch.dtype`` or dict of ``TVTensor`` -> ``torch.dtype``): The dtype to convert to.
            If a ``torch.dtype`` is passed, e.g. ``torch.float32``, only images and videos will be converted
            to that dtype: this is for compatibility with :class:`~torchvision.transforms.v2.ConvertImageDtype`.
            A dict can be passed to specify per-tv_tensor conversions, e.g.
            ``dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others":None}``. The "others"
            key can be used as a catch-all for any other tv_tensor type, and ``None`` means no conversion.
        scale (bool, optional): Whether to scale the values for images or videos. See :ref:`range_and_dtype`.
            Default: ``False``.
    """
    _transformed_types = (torch.Tensor,)

    def __init__(self, dtype: Union[torch.dtype, Dict[Union[Type, str], Optional[torch.dtype]]], scale: bool=False) -> None:
        super().__init__()
        if not isinstance(dtype, (dict, torch.dtype)):
            raise ValueError(f'dtype must be a dict or a torch.dtype, got {type(dtype)} instead')
        if isinstance(dtype, dict) and torch.Tensor in dtype and any((cls in dtype for cls in [tv_tensors.Image, tv_tensors.Video])):
            warnings.warn('Got `dtype` values for `torch.Tensor` and either `tv_tensors.Image` or `tv_tensors.Video`. Note that a plain `torch.Tensor` will *not* be transformed by this (or any other transformation) in case a `tv_tensors.Image` or `tv_tensors.Video` is present in the input.')
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(self.dtype, torch.dtype):
            if not is_pure_tensor(inpt) and (not isinstance(inpt, (tv_tensors.Image, tv_tensors.Video))):
                return inpt
            dtype: Optional[torch.dtype] = self.dtype
        elif type(inpt) in self.dtype:
            dtype = self.dtype[type(inpt)]
        elif 'others' in self.dtype:
            dtype = self.dtype['others']
        else:
            raise ValueError(f"""No dtype was specified for type {type(inpt)}. If you only need to convert the dtype of images or videos, you can just pass e.g. dtype=torch.float32. If you're passing a dict as dtype, you can use "others" as a catch-all key e.g. dtype={{tv_tensors.Mask: torch.int64, "others": None}} to pass-through the rest of the inputs.""")
        supports_scaling = is_pure_tensor(inpt) or isinstance(inpt, (tv_tensors.Image, tv_tensors.Video))
        if dtype is None:
            if self.scale and supports_scaling:
                warnings.warn('scale was set to True but no dtype was specified for images or videos: no scaling will be done.')
            return inpt
        return self._call_kernel(F.to_dtype, inpt, dtype=dtype, scale=self.scale)