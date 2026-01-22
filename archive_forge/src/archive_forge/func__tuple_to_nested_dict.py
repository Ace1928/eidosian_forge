import logging
import re
import sys
import threading
from collections import defaultdict, deque
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Tuple, Union
import requests
import requests.adapters
import urllib3
import wandb
from wandb.sdk.lib import hashutil, telemetry
from .aggregators import aggregate_last, aggregate_mean
from .interfaces import Interface, Metric, MetricsMonitor
def _tuple_to_nested_dict(nested_tuple: Tuple[Tuple[str, Tuple[str, str]], ...]) -> Dict[str, Dict[str, str]]:
    return {k: dict(v) for k, *v in nested_tuple}