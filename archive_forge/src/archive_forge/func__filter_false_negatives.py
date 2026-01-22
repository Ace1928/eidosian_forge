from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _filter_false_negatives(target_areas: Dict[_Color, Tensor], target_segment_matched: Set[_Color], intersection_areas: Dict[Tuple[_Color, _Color], Tensor], void_color: Tuple[int, int]) -> Iterator[int]:
    """Filter false negative segments and yield their category IDs.

    False negatives occur when a ground truth segment is not matched with a prediction.
    Areas that are mostly void in the prediction are ignored.

    Args:
        target_areas: Mapping from colors of the ground truth segments to their extents.
        target_segment_matched: Set of ground truth segments that have been matched to a prediction.
        intersection_areas: Mapping from tuples of `(pred_color, target_color)` to their extent.
        void_color: An additional color that is masked out during metrics calculation.

    Yields:
        Category IDs of segments that account for false negatives.

    """
    false_negative_colors = set(target_areas) - target_segment_matched
    false_negative_colors.discard(void_color)
    for target_color in false_negative_colors:
        void_target_area = intersection_areas.get((void_color, target_color), 0)
        if void_target_area / target_areas[target_color] <= 0.5:
            yield target_color[0]