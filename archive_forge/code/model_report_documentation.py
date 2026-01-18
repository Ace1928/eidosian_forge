from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig

        Generates a QConfigMapping based on the suggestions of the
        ModelReport API for equalization. The generated mapping encompasses all the
        different types of feedback from the input-weight equalization detector.

        These configs are based on the suggestions provided by the ModelReport API
        and can only be generated once the reports have been generated.

        Returns a QConfigMapping for the equalization configuration
        