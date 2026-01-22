import cmath
import math
import warnings
from collections import OrderedDict
from typing import Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple
def _get_builtin_table():
    global _builtin_table
    if _builtin_table is not None:
        return _builtin_table
    _builtin_table = {}

    def register_all(mod):
        for name in dir(mod):
            v = getattr(mod, name)
            if callable(v) and (not _is_special_functional_bound_op(v)) and (v is not torch.no_grad) and (v is not torch.autocast):
                if name == '_segment_reduce':
                    name = name[1:]
                _builtin_ops.append((v, 'aten::' + name))
    for mod in _modules_containing_builtins:
        register_all(mod)
    _builtin_ops.append((math.gcd, 'aten::gcd'))
    _builtin_ops.append((math.isfinite, 'aten::isfinite'))
    _builtin_ops.append((math.remainder, 'aten::mathremainder'))
    import torch.distributed.autograd as dist_autograd
    if dist_autograd.is_available():
        _builtin_ops.append((dist_autograd.get_gradients, 'aten::get_gradients'))
        _builtin_ops.append((dist_autograd.backward, 'aten::dist_backward'))
    for builtin, aten_op in _builtin_ops:
        _builtin_table[id(builtin)] = aten_op
    return _builtin_table