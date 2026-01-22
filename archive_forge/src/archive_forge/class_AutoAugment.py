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
class AutoAugment(_AutoAugmentBase):
    """[BETA] AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.

    .. v2betastatus:: AutoAugment transform

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy, optional): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """
    _v1_transform_cls = _transforms.AutoAugment
    _AUGMENTATION_SPACE = {'ShearX': (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True), 'ShearY': (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True), 'TranslateX': (lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins), True), 'TranslateY': (lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins), True), 'Rotate': (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True), 'Brightness': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Color': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Contrast': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Sharpness': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Posterize': (lambda num_bins, height, width: (8 - torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False), 'Solarize': (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False), 'AutoContrast': (lambda num_bins, height, width: None, False), 'Equalize': (lambda num_bins, height, width: None, False), 'Invert': (lambda num_bins, height, width: None, False)}

    def __init__(self, policy: AutoAugmentPolicy=AutoAugmentPolicy.IMAGENET, interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=None) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.policy = policy
        self._policies = self._get_policies(policy)

    def _get_policies(self, policy: AutoAugmentPolicy) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [(('Posterize', 0.4, 8), ('Rotate', 0.6, 9)), (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)), (('Equalize', 0.8, None), ('Equalize', 0.6, None)), (('Posterize', 0.6, 7), ('Posterize', 0.6, 6)), (('Equalize', 0.4, None), ('Solarize', 0.2, 4)), (('Equalize', 0.4, None), ('Rotate', 0.8, 8)), (('Solarize', 0.6, 3), ('Equalize', 0.6, None)), (('Posterize', 0.8, 5), ('Equalize', 1.0, None)), (('Rotate', 0.2, 3), ('Solarize', 0.6, 8)), (('Equalize', 0.6, None), ('Posterize', 0.4, 6)), (('Rotate', 0.8, 8), ('Color', 0.4, 0)), (('Rotate', 0.4, 9), ('Equalize', 0.6, None)), (('Equalize', 0.0, None), ('Equalize', 0.8, None)), (('Invert', 0.6, None), ('Equalize', 1.0, None)), (('Color', 0.6, 4), ('Contrast', 1.0, 8)), (('Rotate', 0.8, 8), ('Color', 1.0, 2)), (('Color', 0.8, 8), ('Solarize', 0.8, 7)), (('Sharpness', 0.4, 7), ('Invert', 0.6, None)), (('ShearX', 0.6, 5), ('Equalize', 1.0, None)), (('Color', 0.4, 0), ('Equalize', 0.6, None)), (('Equalize', 0.4, None), ('Solarize', 0.2, 4)), (('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)), (('Invert', 0.6, None), ('Equalize', 1.0, None)), (('Color', 0.6, 4), ('Contrast', 1.0, 8)), (('Equalize', 0.8, None), ('Equalize', 0.6, None))]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [(('Invert', 0.1, None), ('Contrast', 0.2, 6)), (('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)), (('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)), (('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)), (('AutoContrast', 0.5, None), ('Equalize', 0.9, None)), (('ShearY', 0.2, 7), ('Posterize', 0.3, 7)), (('Color', 0.4, 3), ('Brightness', 0.6, 7)), (('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)), (('Equalize', 0.6, None), ('Equalize', 0.5, None)), (('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)), (('Color', 0.7, 7), ('TranslateX', 0.5, 8)), (('Equalize', 0.3, None), ('AutoContrast', 0.4, None)), (('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)), (('Brightness', 0.9, 6), ('Color', 0.2, 8)), (('Solarize', 0.5, 2), ('Invert', 0.0, None)), (('Equalize', 0.2, None), ('AutoContrast', 0.6, None)), (('Equalize', 0.2, None), ('Equalize', 0.6, None)), (('Color', 0.9, 9), ('Equalize', 0.6, None)), (('AutoContrast', 0.8, None), ('Solarize', 0.2, 8)), (('Brightness', 0.1, 3), ('Color', 0.7, 0)), (('Solarize', 0.4, 5), ('AutoContrast', 0.9, None)), (('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)), (('AutoContrast', 0.9, None), ('Solarize', 0.8, 3)), (('Equalize', 0.8, None), ('Invert', 0.1, None)), (('TranslateY', 0.7, 9), ('AutoContrast', 0.9, None))]
        elif policy == AutoAugmentPolicy.SVHN:
            return [(('ShearX', 0.9, 4), ('Invert', 0.2, None)), (('ShearY', 0.9, 8), ('Invert', 0.7, None)), (('Equalize', 0.6, None), ('Solarize', 0.6, 6)), (('Invert', 0.9, None), ('Equalize', 0.6, None)), (('Equalize', 0.6, None), ('Rotate', 0.9, 3)), (('ShearX', 0.9, 4), ('AutoContrast', 0.8, None)), (('ShearY', 0.9, 8), ('Invert', 0.4, None)), (('ShearY', 0.9, 5), ('Solarize', 0.2, 6)), (('Invert', 0.9, None), ('AutoContrast', 0.8, None)), (('Equalize', 0.6, None), ('Rotate', 0.9, 3)), (('ShearX', 0.9, 4), ('Solarize', 0.3, 3)), (('ShearY', 0.8, 8), ('Invert', 0.7, None)), (('Equalize', 0.9, None), ('TranslateY', 0.6, 6)), (('Invert', 0.9, None), ('Equalize', 0.6, None)), (('Contrast', 0.3, 3), ('Rotate', 0.8, 4)), (('Invert', 0.8, None), ('TranslateY', 0.0, 2)), (('ShearY', 0.7, 6), ('Solarize', 0.4, 8)), (('Invert', 0.6, None), ('Rotate', 0.8, 4)), (('ShearY', 0.3, 7), ('TranslateX', 0.9, 3)), (('ShearX', 0.1, 6), ('Invert', 0.6, None)), (('Solarize', 0.7, 2), ('TranslateY', 0.6, 7)), (('ShearY', 0.8, 4), ('Invert', 0.8, None)), (('ShearX', 0.7, 9), ('TranslateY', 0.8, 3)), (('ShearY', 0.8, 5), ('AutoContrast', 0.7, None)), (('ShearX', 0.7, 2), ('Invert', 0.1, None))]
        else:
            raise ValueError(f'The provided policy {policy} is not recognized.')

    def forward(self, *inputs: Any) -> Any:
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)
        policy = self._policies[int(torch.randint(len(self._policies), ()))]
        for transform_id, probability, magnitude_idx in policy:
            if not torch.rand(()) <= probability:
                continue
            magnitudes_fn, signed = self._AUGMENTATION_SPACE[transform_id]
            magnitudes = magnitudes_fn(10, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[magnitude_idx])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0
            image_or_video = self._apply_image_or_video_transform(image_or_video, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill)
        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)