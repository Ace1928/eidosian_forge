from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
class FullGraphNode(Node):
    __slots__ = ['value', 'recipe']

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.value = value
        self.recipe = (fun, args, kwargs, zip(parent_argnums, parents))

    def initialize_root(self):
        self.value = None
        self.recipe = (lambda x: x, (), {}, [])