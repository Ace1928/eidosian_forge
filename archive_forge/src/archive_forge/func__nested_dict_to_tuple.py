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
def _nested_dict_to_tuple(nested_dict: Mapping[str, Mapping[str, str]]) -> Tuple[Tuple[str, Tuple[str, str]], ...]:
    return tuple(((k, *v.items()) for k, v in nested_dict.items()))