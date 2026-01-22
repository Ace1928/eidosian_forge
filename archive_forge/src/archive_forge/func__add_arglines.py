from __future__ import annotations
import logging # isort:skip
from inspect import Parameter
from ..models import Marker
def _add_arglines(arglines, param, typ, doc):
    default = param.default if param.default != Parameter.empty else None
    arglines.append(f'    {param.name} ({typ}{(', optional' if default else '')}):')
    if doc:
        arglines += [f'    {x}' for x in doc.rstrip().strip('\n').split('\n')]
    if arglines and default is not None:
        arglines[-1] += f' (default: {default!r})'
    arglines.append('')