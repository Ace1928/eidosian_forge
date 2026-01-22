import logging
import multiprocessing
import os
import platform
import queue
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import yaml
import wandb
from wandb import util, wandb_lib, wandb_sdk
from wandb.agents.pyagent import pyagent
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
def _command_stop(self, command):
    run_id = command['run_id']
    if run_id in self._run_processes:
        proc = self._run_processes[run_id]
        now = util.stopwatch_now()
        if proc.last_sigterm_time is None:
            proc.last_sigterm_time = now
            logger.info('Stop: %s', run_id)
            try:
                proc.terminate()
            except OSError:
                pass
        elif now > proc.last_sigterm_time + self._kill_delay:
            logger.info('Kill: %s', run_id)
            try:
                proc.kill()
            except OSError:
                pass
    else:
        logger.error('Run %s not running', run_id)