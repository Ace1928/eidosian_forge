import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
@staticmethod
def _get_title_series(name):
    for prefix in ['eval', 'test', 'train']:
        if name.startswith(prefix + '_'):
            return (name[len(prefix) + 1:], prefix)
    return (name, 'train')