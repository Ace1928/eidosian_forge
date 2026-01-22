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