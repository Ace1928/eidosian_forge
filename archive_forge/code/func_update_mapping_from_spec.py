import collections
import contextlib
import doctest
import functools
import importlib
import inspect
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Union
from unittest import mock
from unittest.mock import patch
import urllib3
from transformers import logging as transformers_logging
from .integrations import (
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
import asyncio  # noqa
def update_mapping_from_spec(device_fn_dict: Dict[str, Callable], attribute_name: str):
    try:
        spec_fn = getattr(device_spec_module, attribute_name)
        device_fn_dict[torch_device] = spec_fn
    except AttributeError as e:
        if 'default' not in device_fn_dict:
            raise AttributeError(f"`{attribute_name}` not found in '{device_spec_path}' and no default fallback function found.") from e