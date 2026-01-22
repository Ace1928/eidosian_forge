import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
def _route_prefix_format(cls, v):
    """
    The route_prefix
    1. must start with a / character
    2. must not end with a / character (unless the entire prefix is just /)
    3. cannot contain wildcards (must not have "{" or "}")
    """
    if v is None:
        return v
    if not v.startswith('/'):
        raise ValueError(f'Got "{v}" for route_prefix. Route prefix must start with "/".')
    if len(v) > 1 and v.endswith('/'):
        raise ValueError(f'Got "{v}" for route_prefix. Route prefix cannot end with "/" unless the entire prefix is just "/".')
    if '{' in v or '}' in v:
        raise ValueError(f'Got "{v}" for route_prefix. Route prefix cannot contain wildcards, so it cannot contain "{{" or "}}".')
    return v