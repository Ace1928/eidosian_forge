import asyncio
import json
import time
from dataclasses import dataclass, replace, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from ray._private import ray_constants
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.packaging import parse_uri
from ray.experimental.internal_kv import (
from ray.util.annotations import PublicAPI
def uri_to_http_components(package_uri: str) -> Tuple[str, str]:
    suffix = Path(package_uri).suffix
    if suffix not in {'.zip', '.whl'}:
        raise ValueError(f'package_uri ({package_uri}) does not end in .zip or .whl')
    protocol, package_name = parse_uri(package_uri)
    return (protocol.value, package_name)