import inspect
import os
import shutil
import sys
from collections import defaultdict
from inspect import Parameter, Signature
from pathlib import Path
from types import FunctionType
import param
from pyviz_comms import extension as _pyviz_extension
from ..core import (
from ..core.operation import Operation, OperationCallable
from ..core.options import Keywords, Options, options_policy
from ..core.overlay import Overlay
from ..core.util import merge_options_to_dict
from ..operation.element import function
from ..streams import Params, Stream, streams_list_from_dict
from .settings import OutputSettings, list_backends, list_formats
@classmethod
def _group_kwargs_to_options(cls, obj, kwargs):
    """Format option group kwargs into canonical options format"""
    groups = Options._option_groups
    if set(kwargs.keys()) - set(groups):
        raise Exception('Keyword options {} must be one of  {}'.format(groups, ','.join((repr(g) for g in groups))))
    elif not all((isinstance(v, dict) for v in kwargs.values())):
        raise Exception('The %s options must be specified using dictionary groups' % ','.join((repr(k) for k in kwargs.keys())))
    targets = [grp and all((k[0].isupper() for k in grp)) for grp in kwargs.values()]
    if any(targets) and (not all(targets)):
        raise Exception("Cannot mix target specification keys such as 'Image' with non-target keywords.")
    elif not any(targets):
        sanitized_group = util.group_sanitizer(obj.group)
        if obj.label:
            identifier = '{}.{}.{}'.format(obj.__class__.__name__, sanitized_group, util.label_sanitizer(obj.label))
        elif sanitized_group != obj.__class__.__name__:
            identifier = f'{obj.__class__.__name__}.{sanitized_group}'
        else:
            identifier = obj.__class__.__name__
        options = {identifier: {grp: kws for grp, kws in kwargs.items()}}
    else:
        dfltdict = defaultdict(dict)
        for grp, entries in kwargs.items():
            for identifier, kws in entries.items():
                dfltdict[identifier][grp] = kws
        options = dict(dfltdict)
    return options