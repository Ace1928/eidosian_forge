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
def _get_submod_inputs(self, main_module: torch.fx.GraphModule, submod_path: str) -> Tuple[Tensors, Tensors]:
    """
        Try get submodule inputs from stored outputs. If not found then use
        torch_glow.get_submod_inputs to get the inputs.

        If accumulate_error is False, use a_input for run_a() and run_b()
        otherwise use a_input for run_a and b_input for run_b.

        Args:
            main_module: Top-levlel fx module.
            submod_path: Path to the submodule we want to run and compare results.

        Returns:
            a_input: List of tensor(s) that will be used by run_a() as submodule inputs.
            b_input: List of tensor(s) that will be used by run_b() as submodule inputs.
        """
    a_input = []
    b_input = []
    submodule = getattr(main_module, submod_path)
    placeholders = [node.name for node in submodule.graph.nodes if node.op == 'placeholder']
    if set(placeholders) <= self.a_outputs.keys():
        for name in placeholders:
            a_input.append(self.a_outputs[name])
            b_input.append(self.b_outputs[name])
    else:
        if self.settings.accumulate_error:
            print(f"Can't find previous stored outputs named {placeholders}!")

        def get_inputs(self: torch.nn.Module, inputs: Any):
            nonlocal a_input
            a_input = inputs
        handle = submodule.register_forward_pre_hook(get_inputs)
        main_module(*self.sample_input)
        handle.remove()
        b_input = a_input
    if not self.settings.accumulate_error:
        return (a_input, a_input)
    return (a_input, b_input)