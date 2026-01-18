import logging
import os
import platform
import sys
import time
import ray  # noqa F401
import psutil  # noqa E402
Helper class for raising errors on low memory.

    This presents a much cleaner error message to users than what would happen
    if we actually ran out of memory.

    The monitor tries to use the cgroup memory limit and usage if it is set
    and available so that it is more reasonable inside containers. Otherwise,
    it uses `psutil` to check the memory usage.

    The environment variable `RAY_MEMORY_MONITOR_ERROR_THRESHOLD` can be used
    to overwrite the default error_threshold setting.

    Used by test only. For production code use memory_monitor.cc
    