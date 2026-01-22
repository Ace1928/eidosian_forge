from typing import Optional, Tuple, Union
from torch import Tensor
from typing_extensions import Literal
def _total_variation_compute(score: Tensor, num_elements: Union[int, Tensor], reduction: Optional[Literal['mean', 'sum', 'none']]) -> Tensor:
    """Compute final total variation score."""
    if reduction == 'mean':
        return score.sum() / num_elements
    if reduction == 'sum':
        return score.sum()
    if reduction is None or reduction == 'none':
        return score
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")