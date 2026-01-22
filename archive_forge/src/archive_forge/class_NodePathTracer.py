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
class NodePathTracer(LeafModuleAwareTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    name of the Node from which the operation originated. A node name here is
    a `.` separated path walking the hierarchy from top level module down to
    leaf operation or leaf module. The name of the top level module is not
    included as part of the node name. For example, if we trace a module whose
    forward method applies a ReLU module, the name for that node will simply
    be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its node name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate node name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_module_qualname = ''
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `fx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy('call_module', module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(self, kind: str, target: fx.node.Target, args, kwargs, name=None, type_expr=None, *_) -> fx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(self, module_qualname: str, node: fx.node.Node) -> str:
        node_qualname = module_qualname
        if node.op != 'call_module':
            if len(node_qualname) > 0:
                node_qualname += '.'
            node_qualname += str(node)
        if re.match('.+_[0-9]+$', node_qualname) is not None:
            node_qualname = node_qualname.rsplit('_', 1)[0]
        for existing_qualname in reversed(self.node_to_qualname.values()):
            if re.match(f'{node_qualname}(_[0-9]+)?$', existing_qualname) is not None:
                postfix = existing_qualname.replace(node_qualname, '')
                if len(postfix):
                    next_index = int(postfix[1:]) + 1
                else:
                    next_index = 1
                node_qualname += f'_{next_index}'
                break
        return node_qualname