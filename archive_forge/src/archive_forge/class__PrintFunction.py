from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef, UndefinedFunction, Function
class _PrintFunction:
    """
    Function wrapper to replace ``**settings`` in the signature with printer defaults
    """

    def __init__(self, f, print_cls: Type[Printer]):
        params = list(inspect.signature(f).parameters.values())
        assert params.pop(-1).kind == inspect.Parameter.VAR_KEYWORD
        self.__other_params = params
        self.__print_cls = print_cls
        update_wrapper(self, f)

    def __reduce__(self):
        return self.__wrapped__.__qualname__

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        settings = self.__print_cls._get_initial_settings()
        return inspect.Signature(parameters=self.__other_params + [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v) for k, v in settings.items()], return_annotation=self.__wrapped__.__annotations__.get('return', inspect.Signature.empty))