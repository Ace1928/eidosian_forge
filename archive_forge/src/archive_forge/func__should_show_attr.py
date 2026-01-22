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
def _should_show_attr(k, v):
    if k.startswith('_'):
        return False
    if k == 'id':
        return False
    if v is None:
        return False
    if isinstance(v, Iterable) and (not isinstance(v, (str, bytes, bytearray))):
        return not all((x is None for x in v))
    if isinstance(v, Layout) and v.x == 0 and (v.y == 0) and (v.w == 8) and (v.h == 6):
        return False
    return True