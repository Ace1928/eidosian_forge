import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def _accumulate_traverse(self, nodes: NodeList) -> NodeSet:
    culprits: NodeSet = set()
    nodes_to_run: NodeSet = set()
    if self.settings.find_all:
        print("'Find All' mode is not supported in accumulate traversal.")
        return culprits
    for node in nodes:
        report: List[str] = []
        self.reports.append(report)
        self.iteration += 1
        report.append(f'Accumulate traverse iteration {self.iteration}.')
        nodes_to_run.add(node)
        node_name = node.name
        if node_name is not None and isinstance(node_name, tuple):
            node_name = node_name[0]
        assert node_name is not None and isinstance(node_name, str), f'minimize: node_name: {node_name}'
        report.append(f'Add node: {node_name}')
        try:
            split_module, submod_name = self._build_submodule(nodes_to_run)
            self._run_and_compare(split_module, submod_name, [node_name])
            self.print_report(report)
        except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
            culprits.add(node)
            report.append(f'Found culprit {node}')
            self.print_report(report)
            return culprits
    return culprits