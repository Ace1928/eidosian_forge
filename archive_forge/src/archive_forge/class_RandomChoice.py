from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch import nn
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import Transform
class RandomChoice(Transform):
    """[BETA] Apply single transformation randomly picked from a list.

    .. v2betastatus:: RandomChoice transform

    This transform does not support torchscript.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (list of floats or None, optional): probability of each transform being picked.
            If ``p`` doesn't sum to 1, it is automatically normalized. If ``None``
            (default), all transforms have the same probability.
    """

    def __init__(self, transforms: Sequence[Callable], p: Optional[List[float]]=None) -> None:
        if not isinstance(transforms, Sequence):
            raise TypeError('Argument transforms should be a sequence of callables')
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}")
        super().__init__()
        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]

    def forward(self, *inputs: Any) -> Any:
        idx = int(torch.multinomial(torch.tensor(self.p), 1))
        transform = self.transforms[idx]
        return transform(*inputs)