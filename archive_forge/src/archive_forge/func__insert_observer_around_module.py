from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def _insert_observer_around_module(self, obs_fqn: str, target_node: torch.fx.node.Node, obs_to_insert: ObserverBase, observer_args: Tuple, insert_post: bool):
    """
        Helper function that inserts the observer into both the graph structure and the module of the model

        Args
            node_fqn (str): The fully qualified name of the observer we want to insert
            target_node (torch.fx.node.Node): The node in model we are inserting observers around
            obs_to_insert (ObserverBase): The observer we are inserting around target_node
            observer_args (Tuple): The arguments we want to pass into the observer
            insert_post (bool): whether this is meant to be a post observer for this node
        """
    if insert_post:
        target_node = target_node.next
    with self._model.graph.inserting_before(target_node):
        self._model.add_submodule(obs_fqn, obs_to_insert)
        self._model.graph.create_node(op='call_module', target=obs_fqn, args=observer_args)
    self._model.recompile()