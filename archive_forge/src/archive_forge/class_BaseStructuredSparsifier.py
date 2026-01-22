from itertools import chain
from operator import getitem
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union
from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param
from .match_utils import apply_match, MatchAllNode
from .prune_functions import (
class BaseStructuredSparsifier(BaseSparsifier):
    """Base class for structured pruning.

    Abstract methods that need to be implemented:
        - update_mask: Function to compute a new mask for all keys in the
            `groups` attribute.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """

    def __init__(self, defaults, patterns=None):
        super().__init__(defaults)
        if patterns is None:
            patterns = _get_default_structured_pruning_patterns()
        self.patterns = patterns

    def make_config_from_model(self, model: nn.Module, SUPPORTED_MODULES: Optional[Set[Type]]=None) -> None:
        if SUPPORTED_MODULES is None:
            SUPPORTED_MODULES = _get_supported_structured_pruning_modules()
        super().make_config_from_model(model, SUPPORTED_MODULES=SUPPORTED_MODULES)

    def _prepare(self, *args, **kwargs) -> None:
        """This function will attach the FakeStructuredSparsity parameterizations
        and BiasHooks at the appropriate points in the model.
        """
        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrization = config.get('parametrization', FakeStructuredSparsity)
            tensor = getattr(module, tensor_name)
            mask = config.get('mask', torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device))
            self.state[config['tensor_fqn']]['mask'] = mask
            parametrize.register_parametrization(module, tensor_name, parametrization(mask))
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune_bias = config.get('prune_bias', True)
                if module.bias is not None:
                    module.register_parameter('_bias', nn.Parameter(module.bias.detach()))
                    module.bias = None
                    module.prune_bias = prune_bias
                module.register_forward_hook(BiasHook(module.parametrizations.weight[0], prune_bias))

    def prune(self) -> None:
        """
        This function will FX symbolically trace the model and then find instances of the patterns
        defined in self.patterns (by default SUPPORTED_STRUCTURED_PRUNING_PATTERNS ).

        For each pattern, it will apply to corresponding conversion function, which will modify the output
        and input size expected by the modules within the pattern
        """
        self.traced = symbolic_trace(self.model)
        modules = dict(self.traced.named_modules())
        for node in self.traced.graph.nodes:
            for pattern, convert_fn in self.patterns.items():
                matched = apply_match(modules, pattern, node, [])
                if matched is None:
                    continue
                first_module = modules.get(node.target)
                if first_module is not None and parametrize.is_parametrized(first_module) and module_contains_param(first_module, FakeStructuredSparsity):
                    convert_block = []
                    for node in matched:
                        if node.op == 'call_module':
                            convert_block.append(modules.get(node.target))
                        elif node.op == 'call_function':
                            convert_block.append(node.target)
                    convert_fn(*convert_block)
        for module in self.traced.modules():
            if module_contains_param(module, FakeStructuredSparsity):
                raise Exception(f'Error: {module} still contains FakeStructuredSparsity parametrizations!')
        self.traced.graph.lint()
        self.traced.recompile()
        return self.traced