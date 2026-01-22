from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _filter_false_positives(pred_areas: Dict[_Color, Tensor], pred_segment_matched: Set[_Color], intersection_areas: Dict[Tuple[_Color, _Color], Tensor], void_color: Tuple[int, int]) -> Iterator[int]:
    """Filter false positive segments and yield their category IDs.

    False positives occur when a predicted segment is not matched with a corresponding target one.
    Areas that are mostly void in the target are ignored.

    Args:
        pred_areas: Mapping from colors of the predicted segments to their extents.
        pred_segment_matched: Set of predicted segments that have been matched to a ground truth.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false positives.

    """
    false_positive_colors = set(pred_areas) - pred_segment_matched
    false_positive_colors.discard(void_color)
    for pred_color in false_positive_colors:
        pred_void_area = intersection_areas.get((pred_color, void_color), 0)
        if pred_void_area / pred_areas[pred_color] <= 0.5:
            yield pred_color[0]