import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def _build_python_executable_command_memory_profileable(component: str, session_dir: str, unbuffered: bool=True):
    """Build the Python executable command.

    It runs a memory profiler if env var is configured.

    Args:
        component: Name of the component. It must be one of
            ray_constants.PROCESS_TYPE*.
        session_dir: The directory name of the Ray session.
        unbuffered: If true, Python executable is started with unbuffered option.
            e.g., `-u`.
            It means the logs are flushed immediately (good when there's a failure),
            but writing to a log file can be slower.
    """
    command = [sys.executable]
    if unbuffered:
        command.append('-u')
    components_to_memory_profile = os.getenv(RAY_MEMRAY_PROFILE_COMPONENT_ENV, '')
    if not components_to_memory_profile:
        return command
    components_to_memory_profile = set(components_to_memory_profile.split(','))
    try:
        import memray
    except ImportError:
        raise ImportError(f'Memray is required to memory profiler on components {components_to_memory_profile}. Run `pip install memray`.')
    if component in components_to_memory_profile:
        session_dir = Path(session_dir)
        session_name = session_dir.name
        profile_dir = session_dir / 'profile'
        profile_dir.mkdir(exist_ok=True)
        output_file_path = profile_dir / f'{session_name}_memory_{component}.bin'
        options = os.getenv(RAY_MEMRAY_PROFILE_OPTIONS_ENV, None)
        options = options.split(',') if options else []
        command.extend(['-m', 'memray', 'run', '-o', str(output_file_path), *options])
    return command