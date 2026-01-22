from typing import Dict, Optional, Tuple
import torch
def _calculate_expected_aligned_error(alignment_confidence_breaks: torch.Tensor, aligned_distance_error_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (torch.sum(aligned_distance_error_probs * bin_centers, dim=-1), bin_centers[-1])