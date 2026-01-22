import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
def _validate_error_prefix(parameter, attribute=None):
    """
    Generate an error prefix suitable for Parameters when they raise a validation
    error.

    - unbound and name can't be found: "Number parameter"
    - unbound and name can be found: "Number parameter 'x'"
    - bound parameter: "Number parameter 'P.x'"
    """
    from param.parameterized import ParameterizedMetaclass
    pclass = type(parameter).__name__
    if parameter.owner is not None:
        if type(parameter.owner) is ParameterizedMetaclass:
            powner = parameter.owner.__name__
        else:
            powner = type(parameter.owner).__name__
    else:
        powner = None
    pname = parameter.name
    out = []
    if attribute:
        out.append(f'Attribute {attribute!r} of')
    out.append(f'{pclass} parameter')
    if pname:
        if powner:
            desc = f'{powner}.{pname}'
        else:
            desc = pname
        out.append(f'{desc!r}')
    else:
        try:
            pname = _find_pname(pclass)
            if pname:
                out.append(f'{pname!r}')
        except Exception:
            pass
    return ' '.join(out)