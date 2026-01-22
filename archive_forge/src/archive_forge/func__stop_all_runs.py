import ctypes
import logging
import os
import queue
import socket
import threading
import time
import traceback
import wandb
from wandb import wandb_sdk
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
def _stop_all_runs(self):
    logger.debug('Stopping all runs.')
    for run in list(self._run_threads.keys()):
        self._stop_run(run)