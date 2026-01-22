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
def _heartbeat(self):
    while True:
        if self._exit_flag:
            return
        run_status = {run: True for run, status in self._run_status.items() if status in (RunStatus.QUEUED, RunStatus.RUNNING)}
        commands = self._api.agent_heartbeat(self._agent_id, {}, run_status)
        if commands:
            job = Job(commands[0])
            logger.debug(f'Job received: {job}')
            if job.type in ['run', 'resume']:
                self._queue.put(job)
                self._run_status[job.run_id] = RunStatus.QUEUED
            elif job.type == 'stop':
                self._stop_run(job.run_id)
            elif job.type == 'exit':
                self._exit()
                return
        time.sleep(5)