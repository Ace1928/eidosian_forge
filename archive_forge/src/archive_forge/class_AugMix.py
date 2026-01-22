import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms import _functional_tensor as _FT
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
from ._utils import _get_fill, _setup_fill_arg, check_type, is_pure_tensor
class AugMix(_AutoAugmentBase):
    """[BETA] AugMix data augmentation method based on
    `"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" <https://arxiv.org/abs/1912.02781>`_.

    .. v2betastatus:: AugMix transform

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        severity (int, optional): The severity of base augmentation operators. Default is ``3``.
        mixture_width (int, optional): The number of augmentation chains. Default is ``3``.
        chain_depth (int, optional): The depth of augmentation chains. A negative value denotes stochastic depth sampled from the interval [1, 3].
            Default is ``-1``.
        alpha (float, optional): The hyperparameter for the probability distributions. Default is ``1.0``.
        all_ops (bool, optional): Use all operations (including brightness, contrast, color and sharpness). Default is ``True``.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """
    _v1_transform_cls = _transforms.AugMix
    _PARTIAL_AUGMENTATION_SPACE = {'ShearX': (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True), 'ShearY': (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True), 'TranslateX': (lambda num_bins, height, width: torch.linspace(0.0, width / 3.0, num_bins), True), 'TranslateY': (lambda num_bins, height, width: torch.linspace(0.0, height / 3.0, num_bins), True), 'Rotate': (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True), 'Posterize': (lambda num_bins, height, width: (4 - torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False), 'Solarize': (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False), 'AutoContrast': (lambda num_bins, height, width: None, False), 'Equalize': (lambda num_bins, height, width: None, False)}
    _AUGMENTATION_SPACE: Dict[str, Tuple[Callable[[int, int, int], Optional[torch.Tensor]], bool]] = {**_PARTIAL_AUGMENTATION_SPACE, 'Brightness': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Color': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Contrast': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Sharpness': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True)}

    def __init__(self, severity: int=3, mixture_width: int=3, chain_depth: int=-1, alpha: float=1.0, all_ops: bool=True, interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=None) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self._PARAMETER_MAX = 10
        if not 1 <= severity <= self._PARAMETER_MAX:
            raise ValueError(f'The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead.')
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.all_ops = all_ops

    def _sample_dirichlet(self, params: torch.Tensor) -> torch.Tensor:
        return torch._sample_dirichlet(params)

    def forward(self, *inputs: Any) -> Any:
        flat_inputs_with_spec, orig_image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(orig_image_or_video)
        if isinstance(orig_image_or_video, torch.Tensor):
            image_or_video = orig_image_or_video
        else:
            image_or_video = F.pil_to_tensor(orig_image_or_video)
        augmentation_space = self._AUGMENTATION_SPACE if self.all_ops else self._PARTIAL_AUGMENTATION_SPACE
        orig_dims = list(image_or_video.shape)
        expected_ndim = 5 if isinstance(orig_image_or_video, tv_tensors.Video) else 4
        batch = image_or_video.reshape([1] * max(expected_ndim - image_or_video.ndim, 0) + orig_dims)
        batch_dims = [batch.size(0)] + [1] * (batch.ndim - 1)
        m = self._sample_dirichlet(torch.tensor([self.alpha, self.alpha], device=batch.device).expand(batch_dims[0], -1))
        combined_weights = self._sample_dirichlet(torch.tensor([self.alpha] * self.mixture_width, device=batch.device).expand(batch_dims[0], -1)) * m[:, 1].reshape([batch_dims[0], -1])
        mix = m[:, 0].reshape(batch_dims) * batch
        for i in range(self.mixture_width):
            aug = batch
            depth = self.chain_depth if self.chain_depth > 0 else int(torch.randint(low=1, high=4, size=(1,)).item())
            for _ in range(depth):
                transform_id, (magnitudes_fn, signed) = self._get_random_item(augmentation_space)
                magnitudes = magnitudes_fn(self._PARAMETER_MAX, height, width)
                if magnitudes is not None:
                    magnitude = float(magnitudes[int(torch.randint(self.severity, ()))])
                    if signed and torch.rand(()) <= 0.5:
                        magnitude *= -1
                else:
                    magnitude = 0.0
                aug = self._apply_image_or_video_transform(aug, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill)
            mix.add_(combined_weights[:, i].reshape(batch_dims) * aug)
        mix = mix.reshape(orig_dims).to(dtype=image_or_video.dtype)
        if isinstance(orig_image_or_video, (tv_tensors.Image, tv_tensors.Video)):
            mix = tv_tensors.wrap(mix, like=orig_image_or_video)
        elif isinstance(orig_image_or_video, PIL.Image.Image):
            mix = F.to_pil_image(mix)
        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, mix)