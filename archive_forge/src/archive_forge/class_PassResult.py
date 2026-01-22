import abc
from collections import namedtuple
from typing import Optional
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
@compatibility(is_backward_compatible=False)
class PassResult(namedtuple('PassResult', ['graph_module', 'modified'])):
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """

    def __new__(cls, graph_module, modified):
        return super().__new__(cls, graph_module, modified)