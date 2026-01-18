import argparse
import errno
import glob
import logging
import logging.handlers
import os
import platform
import re
import shutil
import time
import traceback
from typing import Callable, List, Optional, Set
from ray._raylet import GcsClient
import ray._private.ray_constants as ray_constants
import ray._private.services as services
import ray._private.utils
from ray._private.ray_logging import setup_component_logger
def update_log_filenames(self):
    """Update the list of log files to monitor."""
    monitor_log_paths = []
    monitor_log_paths += glob.glob(f'{self.logs_dir}/worker*[.out|.err]') + glob.glob(f'{self.logs_dir}/java-worker*.log')
    monitor_log_paths += glob.glob(f'{self.logs_dir}/raylet*.err')
    if not self.is_autoscaler_v2:
        monitor_log_paths += glob.glob(f'{self.logs_dir}/monitor.log')
    else:
        monitor_log_paths += glob.glob(f'{self.logs_dir}/events/event_AUTOSCALER.log')
    monitor_log_paths += glob.glob(f'{self.logs_dir}/gcs_server*.err')
    if RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED:
        monitor_log_paths += glob.glob(f'{self.logs_dir}/runtime_env*.log')
    for file_path in monitor_log_paths:
        if os.path.isfile(file_path) and file_path not in self.log_filenames:
            worker_match = WORKER_LOG_PATTERN.match(file_path)
            if worker_match:
                worker_pid = int(worker_match.group(2))
            else:
                worker_pid = None
            job_id = None
            if 'runtime_env' in file_path:
                runtime_env_job_match = RUNTIME_ENV_SETUP_PATTERN.match(file_path)
                if runtime_env_job_match:
                    job_id = runtime_env_job_match.group(1)
            is_err_file = file_path.endswith('err')
            self.log_filenames.add(file_path)
            self.closed_file_infos.append(LogFileInfo(filename=file_path, size_when_last_opened=0, file_position=0, file_handle=None, is_err_file=is_err_file, job_id=job_id, worker_pid=worker_pid))
            log_filename = os.path.basename(file_path)
            logger.info(f'Beginning to track file {log_filename}')