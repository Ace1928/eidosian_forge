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
def _command_exit(self, command):
    logger.info('Received exit command. Killing runs and quitting.')
    for _, proc in self._run_processes.items():
        try:
            proc.kill()
        except OSError:
            pass
    self._running = False