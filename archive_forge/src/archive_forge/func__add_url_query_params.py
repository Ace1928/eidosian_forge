import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def _add_url_query_params(url: str, params: Dict[str, str]) -> str:
    """Add params to the provided url as query parameters.

    If url already contains query parameters, they will be merged with params, with the
    existing query parameters overriding any in params with the same parameter name.

    Args:
        url: The URL to add query parameters to.
        params: The query parameters to add.

    Returns:
        URL with params added as query parameters.
    """
    url = unquote(url)
    parsed_url = urlparse(url)
    base_params = params
    params = dict(parse_qsl(parsed_url.query))
    base_params.update(params)
    base_params.update({k: json.dumps(v) for k, v in base_params.items() if isinstance(v, (bool, dict))})
    encoded_params = urlencode(base_params, doseq=True)
    parsed_url = parsed_url._replace(query=encoded_params)
    return urlunparse(parsed_url)