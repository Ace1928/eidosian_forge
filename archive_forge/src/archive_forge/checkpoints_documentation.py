import logging
import json
import os
from packaging import version
import re
from typing import Any, Dict, Union
import ray
from ray.rllib.utils.serialization import NOT_SERIALIZABLE, serialize_type
from ray.train import Checkpoint
from ray.util import log_once
from ray.util.annotations import PublicAPI
Tries importing msgpack and msgpack_numpy and returns the patched msgpack module.

    Returns None if error is False and msgpack or msgpack_numpy is not installed.
    Raises an error, if error is True and the modules could not be imported.

    Args:
        error: Whether to raise an error if msgpack/msgpack_numpy cannot be imported.

    Returns:
        The `msgpack` module.

    Raises:
        ImportError: If error=True and msgpack/msgpack_numpy is not installed.
    