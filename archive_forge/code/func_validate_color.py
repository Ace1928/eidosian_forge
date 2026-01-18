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
@validator('color')
def validate_color(cls, v):
    if not internal.is_valid_color(v):
        raise ValueError('invalid color, value should be hex, rgb, or rgba')
    return v