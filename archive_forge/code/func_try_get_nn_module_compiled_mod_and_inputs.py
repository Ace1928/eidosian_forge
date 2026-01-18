from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def try_get_nn_module_compiled_mod_and_inputs(*args, **kwargs):
    name = get_nn_module_name_from_kwargs(**kwargs)
    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        return
    test_name = name
    if 'desc' in kwargs:
        test_name = f'{test_name}_{kwargs['desc']}'
    test_name = get_nn_mod_test_name(**kwargs)
    if test_name in EXCLUDE_SCRIPT_MODULES:
        return
    if 'constructor' in kwargs:
        nn_module = kwargs['constructor']
    else:
        nn_module = getattr(torch.nn, name)
    if 'FunctionalModule' in str(nn_module):
        return
    if 'constructor_args_fn' in kwargs:
        constructor_args = kwargs['constructor_args_fn']()
    else:
        constructor_args = kwargs.get('constructor_args', ())
    input_dtype = torch.double
    if 'input_fn' in kwargs:
        input = kwargs['input_fn']()
        if isinstance(input, torch.Tensor):
            input = (input,)
        if all((tensor.is_complex() for tensor in input)):
            input_dtype = torch.cdouble
    else:
        input = (kwargs['input_size'],)
    if 'extra_args' in kwargs:
        input = input + kwargs['extra_args']
    if 'target_size' in kwargs:
        input = input + (kwargs['target_size'],)
    elif 'target_fn' in kwargs:
        if torch.is_tensor(input):
            input = (input,)
        input = input + (kwargs['target_fn'](),)
    args_variable, kwargs_variable = create_input(input, dtype=input_dtype)
    f_args_variable = deepcopy(unpack_variables(args_variable))
    out_var = deepcopy(f_args_variable)
    args, mod = (f_args_variable, create_script_module(None, nn_module, constructor_args, *f_args_variable)(*f_args_variable))
    return (mod, out_var)