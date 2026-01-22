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
def _generate_info_dict(self, model: GraphModule) -> Dict[str, Dict]:
    """
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns to:
            whether there were outliers found in activation before
            the number of batches used for each channel
            whether fraction of applicable batches used is above fraction_batches_used_threshold
            their p_r metric compared to the threshold
            the threshold used to make the recommendation
            the reference_percentile used to make the recommendation
            the channel axis used to determine individual channels
            the constant batch counts per channel
            the per channel max values
        """
    info_dict: Dict[str, Dict] = {}
    for fqn, module in model.named_modules():
        if self._supports_report_gen(module):
            pre_obs: ModelReportObserver = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            num_batches: torch.Tensor = pre_obs.percentile_batches_tracked
            average_ratios: torch.Tensor = pre_obs.average_percentile_ratio
            channel_batch_cnts: torch.Tensor = pre_obs.constant_channels
            total_batches: int = pre_obs.num_batches_tracked
            max_vals: torch.Tensor = pre_obs.max_val
            for index, ratio_val in enumerate(average_ratios):
                if ratio_val.item() < 0:
                    average_ratios[index] = -ratio_val
                if ratio_val.item() < 1:
                    average_ratios[index] = 1 / ratio_val
            outlier_calcs = self._calculate_outlier_info(average_ratios, num_batches, total_batches)
            info_dict[fqn] = {self.CHANNEL_AXIS_KEY: self.ch_axis, self.REF_PERCENTILE_KEY: self.reference_percentile, self.RATIO_THRES_KEY: self.ratio_threshold, self.COMP_METRIC_KEY: average_ratios, self.NUM_BATCHES_KEY: num_batches, self.OUTLIER_KEY: outlier_calcs[self.OUTLIER_KEY], self.IS_SUFFICIENT_BATCHES_KEY: outlier_calcs[self.IS_SUFFICIENT_BATCHES_KEY], self.CONSTANT_COUNTS_KEY: channel_batch_cnts, self.MAX_VALS_KEY: max_vals}
    return info_dict