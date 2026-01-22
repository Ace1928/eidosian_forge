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
def _metric_to_backend_panel_grid(x: Optional[MetricType]):
    if isinstance(x, str):
        name, *rest = x.split('.')
        rest = '.' + '.'.join(rest) if rest else ''
        return f'config:{name}.value{rest}'
    return _metric_to_backend(x)