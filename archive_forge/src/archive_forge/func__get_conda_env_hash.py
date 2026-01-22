import hashlib
import json
import logging
import os
import platform
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from filelock import FileLock
import ray
from ray._private.runtime_env.conda_utils import (
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.validation import parse_and_validate_conda
from ray._private.utils import (
def _get_conda_env_hash(conda_dict: Dict) -> str:
    serialized_conda_spec = json.dumps(conda_dict, sort_keys=True)
    hash = hashlib.sha1(serialized_conda_spec.encode('utf-8')).hexdigest()
    return hash