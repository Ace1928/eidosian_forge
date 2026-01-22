import functools
import logging
import math
import operator
import sympy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp
from torch.fx.experimental import _config as config
class PopulateValidator(torch.fx.Interpreter):

    def __init__(self, graph: torch.fx.Graph, validator: 'TranslationValidator'):
        self.validator = validator
        module = torch.fx.GraphModule(root={}, graph=graph)
        super().__init__(module, garbage_collect_values=True)

    def placeholder(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        symbol = fx_traceback.get_current_meta()['symbol']
        return self.validator.z3var(symbol)

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        if target != torch._assert:
            return super().call_function(target, args, kwargs)
        assert len(args) == 1, f'expected 1 argument on assertion. Got: {len(args)} '
        self.validator.add_source_expr(args[0])