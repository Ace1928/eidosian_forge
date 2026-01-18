import os.path
import subprocess
import sys
import time
import shutil
import fcntl
import signal
import socket
import logging
import threading
from ray.util.spark.cluster_init import (
from ray._private.ray_process_reaper import SIGTERM_GRACE_PERIOD_SECONDS
def try_clean_temp_dir_at_exit():
    try:
        time.sleep(SIGTERM_GRACE_PERIOD_SECONDS + 0.5)
        if process.poll() is None:
            process.kill()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError:
            lock_acquired = False
        if lock_acquired:
            if collect_log_to_path:
                try:
                    base_dir = os.path.join(collect_log_to_path, os.path.basename(temp_dir) + '-logs')
                    os.makedirs(base_dir, exist_ok=True)
                    copy_log_dest_path = os.path.join(base_dir, socket.gethostname())
                    ray_session_dir = os.readlink(os.path.join(temp_dir, 'session_latest'))
                    shutil.copytree(os.path.join(ray_session_dir, 'logs'), copy_log_dest_path)
                except Exception as e:
                    _logger.warning(f'Collect logs to destination directory failed, error: {repr(e)}.')
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)