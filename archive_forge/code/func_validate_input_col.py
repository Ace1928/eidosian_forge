import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def validate_input_col(fn: Callable, input_col: Optional[Union[int, tuple, list]]):
    """
    Check that function used in a callable datapipe works with the input column.

    This simply ensures that the number of positional arguments matches the size
    of the input column. The function must not contain any non-default
    keyword-only arguments.

    Examples:
        >>> # xdoctest: +SKIP("Failing on some CI machines")
        >>> def f(a, b, *, c=1):
        >>>     return a + b + c
        >>> def f_def(a, b=1, *, c=1):
        >>>     return a + b + c
        >>> assert validate_input_col(f, [1, 2])
        >>> assert validate_input_col(f_def, 1)
        >>> assert validate_input_col(f_def, [1, 2])

    Notes:
        If the function contains variable positional (`inspect.VAR_POSITIONAL`) arguments,
        for example, f(a, *args), the validator will accept any size of input column
        greater than or equal to the number of positional arguments.
        (in this case, 1).

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Raises:
        ValueError: If the function is not compatible with the input column.
    """
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return
    if isinstance(input_col, (list, tuple)):
        input_col_size = len(input_col)
    else:
        input_col_size = 1
    pos = []
    var_positional = False
    non_default_kw_only = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            pos.append(p)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            if p.default is p.empty:
                non_default_kw_only.append(p)
        else:
            continue
    if isinstance(fn, functools.partial):
        fn_name = getattr(fn.func, '__name__', repr(fn.func))
    else:
        fn_name = getattr(fn, '__name__', repr(fn))
    if len(non_default_kw_only) > 0:
        raise ValueError(f'The function {fn_name} takes {len(non_default_kw_only)} non-default keyword-only parameters, which is not allowed.')
    if len(sig.parameters) < input_col_size:
        if not var_positional:
            raise ValueError(f'The function {fn_name} takes {len(sig.parameters)} parameters, but {input_col_size} are required.')
    elif len(pos) > input_col_size:
        if any((p.default is p.empty for p in pos[input_col_size:])):
            raise ValueError(f'The function {fn_name} takes {len(pos)} positional parameters, but {input_col_size} are required.')
    elif len(pos) < input_col_size:
        if not var_positional:
            raise ValueError(f'The function {fn_name} takes {len(pos)} positional parameters, but {input_col_size} are required.')