import importlib
import inspect
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import ray.util.client_connect
from ray._private.ray_constants import (
from ray._private.utils import check_ray_client_dependencies_installed, split_address
from ray._private.worker import BaseContext
from ray._private.worker import init as ray_driver_init
from ray.job_config import JobConfig
from ray.util.annotations import Deprecated, PublicAPI
def _split_address(address: str) -> Tuple[str, str]:
    """
    Splits address into a module string (scheme) and an inner_address.

    If the scheme is not present, then "ray://" is prepended to the address.
    """
    if '://' not in address:
        address = 'ray://' + address
    return split_address(address)