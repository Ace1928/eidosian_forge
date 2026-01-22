from ._ops import OpOverload
from typing import Any, Optional, Set, List
import traceback
import torch
import weakref
import functools
import inspect
import re
import sys
def _check_pystubs_once(func, qualname, actual_module_name):
    checked = False

    def inner(*args, **kwargs):
        nonlocal checked
        if checked:
            return func(*args, **kwargs)
        op = torch._library.utils.lookup_op(qualname)
        if op._defined_in_python:
            checked = True
            return func(*args, **kwargs)
        maybe_pystub = torch._C._dispatch_pystub(op._schema.name, op._schema.overload_name)
        if not maybe_pystub:
            raise RuntimeError(f'''Operator '{qualname}' was defined in C++ and has a Python abstract impl. In this situation, it is required to have a C++ `m.impl_abstract_pystub` call, but we could not find one.Please add a call to `m.impl_abstract_pystub("{actual_module_name}");` to the C++ TORCH_LIBRARY block the operator was defined in.''')
        pystub_module = maybe_pystub[0]
        if actual_module_name != pystub_module:
            raise RuntimeError(f"Operator '{qualname}' specified that its python abstract impl is in the Python module '{pystub_module}' but it was actually found in '{actual_module_name}'. Please either move the abstract impl or correct the m.impl_abstract_pystub call.")
        checked = True
        return func(*args, **kwargs)
    return inner