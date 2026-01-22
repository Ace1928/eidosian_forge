from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _update_detector_equalization_qconfig_info(self, combined_info: DetectorQConfigInfo, new_info: DetectorQConfigInfo):
    """
        Takes in the old and new information and updates the combined information.

        Args:
            combined_info (DetectorQConfigInfo): The DetectorQConfigInfo we are compiling all of the information in
            new_info (DetectorQConfigInfo): The DetectorQConfigInfo with the information we are trying to merge the new info
                into it
        """
    is_equalization_recommended = combined_info.is_equalization_recommended or new_info.is_equalization_recommended
    combined_info.is_equalization_recommended = is_equalization_recommended