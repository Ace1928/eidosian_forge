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
def _metric_to_backend_pc(x: Optional[ParallelCoordinatesMetric]):
    if x is None:
        return x
    if isinstance(x, str):
        name = x
        return f'summary:{name}'
    if isinstance(x, Config):
        name = x.name
        return f'c::{name}'
    if isinstance(x, SummaryMetric):
        name = x.name
        return f'summary:{name}'
    raise Exception('Unexpected metric type')