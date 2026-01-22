import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
def _internal_children_to_text(children):
    pieces = []
    for x in children:
        t = _generate_thing(x)
        if isinstance(t, list):
            for x in t:
                pieces.append(x)
        else:
            pieces.append(t)
    if not pieces:
        return ''
    if len(pieces) == 1 and isinstance(pieces[0], str):
        return pieces[0]
    if len(pieces) == 3 and pieces[0] == '' and (pieces[-1] == ''):
        return pieces[1]
    if len(pieces) >= 3 and pieces[0] == '' and (pieces[-1] == ''):
        return pieces[1:-1]
    if all((x == '' for x in pieces)):
        return ''
    return pieces