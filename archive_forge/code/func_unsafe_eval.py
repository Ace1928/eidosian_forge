from __future__ import annotations
import builtins
from types import CodeType
from typing import Any, Callable
from . import Image, _imagingmath
from ._deprecate import deprecate
def unsafe_eval(expression: str, options: dict[str, Any]={}, **kw: Any) -> Any:
    """
    Evaluates an image expression. This uses Python's ``eval()`` function to process
    the expression string, and carries the security risks of doing so. It is not
    recommended to process expressions without considering this.
    :py:meth:`~lambda_eval` is a more secure alternative.

    :py:mod:`~PIL.ImageMath` only supports single-layer images. To process multi-band
    images, use the :py:meth:`~PIL.Image.Image.split` method or
    :py:func:`~PIL.Image.merge` function.

    :param expression: A string containing a Python-style expression.
    :param options: Values to add to the evaluation context.  You
                    can either use a dictionary, or one or more keyword
                    arguments.
    :return: The evaluated expression. This is usually an image object, but can
             also be an integer, a floating point value, or a pixel tuple,
             depending on the expression.
    """
    args: dict[str, Any] = ops.copy()
    for k in list(options.keys()) + list(kw.keys()):
        if '__' in k or hasattr(builtins, k):
            msg = f"'{k}' not allowed"
            raise ValueError(msg)
    args.update(options)
    args.update(kw)
    for k, v in args.items():
        if hasattr(v, 'im'):
            args[k] = _Operand(v)
    compiled_code = compile(expression, '<string>', 'eval')

    def scan(code: CodeType) -> None:
        for const in code.co_consts:
            if type(const) is type(compiled_code):
                scan(const)
        for name in code.co_names:
            if name not in args and name != 'abs':
                msg = f"'{name}' not allowed"
                raise ValueError(msg)
    scan(compiled_code)
    out = builtins.eval(expression, {'__builtins': {'abs': abs}}, args)
    try:
        return out.im
    except AttributeError:
        return out