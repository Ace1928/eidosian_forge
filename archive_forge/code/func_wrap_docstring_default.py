import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
def wrap_docstring_default(r_func: SignatureTranslatedFunction, is_method: bool, signature: inspect.Signature, r_ellipsis: typing.Optional[int], *, full_repr: bool=False) -> str:
    """
    Create a docstring that for a wrapped function.

    Args:
        r_func (SignatureTranslatedFunction): an R function
        is_method (bool): Whether the function should be treated as a method
            (a `self` parameter is added to the signature if so).
        signature (inspect.Signature): A mapped signature for `r_func`
        r_ellipsis (bool): Index of the parameter containing the R ellipsis
            (`...`). None if the R ellipsis is not in the function signature.
        full_repr (bool): Whether to have the full body of the R function in
            the docstring dynamically generated.
    Returns:
        A string.
    """
    docstring = []
    docstring.append('This {} wraps the following R function.'.format('method' if is_method else 'function'))
    if r_ellipsis:
        docstring.extend(('', textwrap.dedent('The R ellipsis "..." present in the function\'s parameters\n                 is mapped to a python iterable of (name, value) pairs (such as\n                 it is returned by the `dict` method `items()` for example.'), ''))
    if full_repr:
        docstring.append('\n{}'.format(r_func.r_repr()))
    else:
        r_repr = r_func.r_repr()
        i = r_repr.find('\n{')
        if i == -1:
            docstring.append('\n{}'.format(r_func.r_repr()))
        else:
            docstring.append('\n{}\n{{\n  ...\n}}'.format(r_repr[:i]))
    return '\n'.join(docstring)