from typing import Optional, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
def _total_variation_update(img: Tensor) -> Tuple[Tensor, int]:
    """Compute total variation statistics on current batch."""
    if img.ndim != 4:
        raise RuntimeError(f'Expected input `img` to be an 4D tensor, but got {img.shape}')
    diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]
    res1 = diff1.abs().sum([1, 2, 3])
    res2 = diff2.abs().sum([1, 2, 3])
    score = res1 + res2
    return (score, img.shape[0])