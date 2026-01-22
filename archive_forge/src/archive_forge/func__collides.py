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
def _collides(p1: Panel, p2: Panel) -> bool:
    l1, l2 = (p1.layout, p2.layout)
    if p1.id == p2.id or l1.x + l1.w <= l2.x or l1.x >= l2.w + l2.x or (l1.y + l1.h <= l2.y) or (l1.y >= l2.y + l2.h):
        return False
    return True