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
def _get_ray_setup_spec():
    """Find the Ray setup_spec from the currently running Ray.

    This function works even when Ray is built from source with pip install -e.
    """
    ray_source_python_path = _resolve_current_ray_path()
    setup_py_path = os.path.join(ray_source_python_path, 'setup.py')
    return runpy.run_path(setup_py_path)['setup_spec']