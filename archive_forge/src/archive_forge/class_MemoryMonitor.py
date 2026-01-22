import logging
import os
import platform
import sys
import time
import ray  # noqa F401
import psutil  # noqa E402
class MemoryMonitor:
    """Helper class for raising errors on low memory.

    This presents a much cleaner error message to users than what would happen
    if we actually ran out of memory.

    The monitor tries to use the cgroup memory limit and usage if it is set
    and available so that it is more reasonable inside containers. Otherwise,
    it uses `psutil` to check the memory usage.

    The environment variable `RAY_MEMORY_MONITOR_ERROR_THRESHOLD` can be used
    to overwrite the default error_threshold setting.

    Used by test only. For production code use memory_monitor.cc
    """

    def __init__(self, error_threshold=0.95, check_interval=1):
        self.check_interval = check_interval
        self.last_checked = 0
        try:
            self.error_threshold = float(os.getenv('RAY_MEMORY_MONITOR_ERROR_THRESHOLD'))
        except (ValueError, TypeError):
            self.error_threshold = error_threshold
        try:
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'rb') as f:
                self.cgroup_memory_limit_gb = int(f.read()) / 1024 ** 3
        except IOError:
            self.cgroup_memory_limit_gb = sys.maxsize / 1024 ** 3
        if not psutil:
            logger.warn('WARNING: Not monitoring node memory since `psutil` is not installed. Install this with `pip install psutil` to enable debugging of memory-related crashes.')
        self.disabled = 'RAY_DEBUG_DISABLE_MEMORY_MONITOR' in os.environ or 'RAY_DISABLE_MEMORY_MONITOR' in os.environ

    def get_memory_usage(self):
        psutil_mem = psutil.virtual_memory()
        total_gb = psutil_mem.total / 1024 ** 3
        used_gb = psutil_mem.used / 1024 ** 3
        if self.cgroup_memory_limit_gb < total_gb:
            total_gb = self.cgroup_memory_limit_gb
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'rb') as f:
                used_gb = int(f.read()) / 1024 ** 3
            with open('/sys/fs/cgroup/memory/memory.stat', 'r') as f:
                for line in f.readlines():
                    if line.split(' ')[0] == 'cache':
                        used_gb = used_gb - int(line.split(' ')[1]) / 1024 ** 3
            assert used_gb >= 0
        return (used_gb, total_gb)

    def raise_if_low_memory(self):
        if self.disabled:
            return
        if time.time() - self.last_checked > self.check_interval:
            self.last_checked = time.time()
            used_gb, total_gb = self.get_memory_usage()
            if used_gb > total_gb * self.error_threshold:
                raise RayOutOfMemoryError(RayOutOfMemoryError.get_message(used_gb, total_gb, self.error_threshold))
            else:
                logger.debug(f'Memory usage is {used_gb} / {total_gb}')