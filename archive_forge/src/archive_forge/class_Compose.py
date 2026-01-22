from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch import nn
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform
class Compose(Transform):
    """[BETA] Composes several transforms together.

    .. v2betastatus:: Compose transform

    This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError('Argument transforms should be a sequence of callables')
        elif not transforms:
            raise ValueError('Pass at least one transform')
        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1
        for transform in self.transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f'    {t}')
        return '\n'.join(format_string)