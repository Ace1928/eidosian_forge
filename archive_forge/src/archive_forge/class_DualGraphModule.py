import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _copy_attr
class DualGraphModule(fx.GraphModule):
    """
    A derivative of `fx.GraphModule`. Differs in the following ways:
    - Requires a train and eval version of the underlying graph
    - Copies submodules according to the nodes of both train and eval graphs.
    - Calling train(mode) switches between train graph and eval graph.
    """

    def __init__(self, root: torch.nn.Module, train_graph: fx.Graph, eval_graph: fx.Graph, class_name: str='GraphModule'):
        """
        Args:
            root (nn.Module): module from which the copied module hierarchy is
                built
            train_graph (fx.Graph): the graph that should be used in train mode
            eval_graph (fx.Graph): the graph that should be used in eval mode
        """
        super(fx.GraphModule, self).__init__()
        self.__class__.__name__ = class_name
        self.train_graph = train_graph
        self.eval_graph = eval_graph
        for node in chain(iter(train_graph.nodes), iter(eval_graph.nodes)):
            if node.op in ['get_attr', 'call_module']:
                if not isinstance(node.target, str):
                    raise TypeError(f'node.target should be of type str instead of {type(node.target)}')
                _copy_attr(root, self, node.target)
        self.train()
        self.graph = train_graph
        if self.eval_graph._tracer_cls != self.train_graph._tracer_cls:
            raise TypeError(f'Train mode and eval mode should use the same tracer class. Instead got {self.eval_graph._tracer_cls} for eval vs {self.train_graph._tracer_cls} for train')
        self._tracer_cls = None
        if self.graph._tracer_cls and '<locals>' not in self.graph._tracer_cls.__qualname__:
            self._tracer_cls = self.graph._tracer_cls

    def train(self, mode=True):
        """
        Swap out the graph depending on the selected training mode.
        NOTE this should be safe when calling model.eval() because that just
        calls this with mode == False.
        """
        if mode and (not self.training):
            self.graph = self.train_graph
        elif not mode and self.training:
            self.graph = self.eval_graph
        return super().train(mode=mode)