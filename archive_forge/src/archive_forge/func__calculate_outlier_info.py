from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.fx._equalize import (
from torch.ao.quantization.observer import _is_activation_post_process
def _calculate_outlier_info(self, percentile_ratios: torch.Tensor, counted_batches: torch.Tensor, total_batches: int) -> Dict[str, List[bool]]:
    """
        Gives info on whether the percentile ratios calculated would be considered outliers
        Also gives information on whether the collected data is statistically significant to make this claim

        Args:
            percentile_ratios (torch.Tensor): The average percentile_ratios per channel calculated by the observer
            counted_batches (torch.Tensor): The number of batches used for average calculation per tensor
            total_batches (int): The total number of batches that passed through observer in this epoch

        Returns a dictionary mapping:
            "outliers_detected" : list of bools per channel that are true if it is considered an outlier
            "is_sufficient_batches": if o_r was >= fraction_batches_used_threshold:
                where o_r = counted_batches / total_batches
        """
    outlier_dict: Dict[str, List[bool]] = {self.OUTLIER_KEY: [], self.IS_SUFFICIENT_BATCHES_KEY: []}
    ratios_list: List = percentile_ratios.tolist()
    num_batches_list: List = counted_batches.tolist()
    significant_size = [batch_size / total_batches >= self.fraction_batches_used_threshold for batch_size in num_batches_list]
    outlier_dict[self.IS_SUFFICIENT_BATCHES_KEY] = significant_size
    outlier_detected = [ratio > self.ratio_threshold for ratio in ratios_list]
    outlier_dict[self.OUTLIER_KEY] = outlier_detected
    return outlier_dict