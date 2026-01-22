from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _get_node_from_fqn(self, node_fqn: str) -> torch.fx.node.Node:
    """
        Takes in a node fqn and returns the node based on the fqn

        Args
            node_fqn (str): The fully qualified name of the node we want to find in model

        Returns the Node object of the given node_fqn otherwise returns None
        """
    node_to_return = None
    for node in self._model.graph.nodes:
        if node.target == node_fqn:
            node_to_return = node
            break
    if node_to_return is None:
        raise ValueError('The node_fqn is was not found within the module.')
    assert isinstance(node_to_return, torch.fx.node.Node)
    return node_to_return