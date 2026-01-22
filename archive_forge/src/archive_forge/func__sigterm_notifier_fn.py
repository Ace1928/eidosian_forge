import logging
import os
import re
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS, _PYTHON_GREATER_EQUAL_3_8_0
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_info
def _sigterm_notifier_fn(self, signum: _SIGNUM, _: FrameType) -> None:
    log.info(rank_prefixed_message(f'Received SIGTERM: {signum}', self.trainer.local_rank))
    if not self.received_sigterm:
        launcher = self.trainer.strategy.launcher
        if launcher is not None:
            launcher.kill(signum)
    self.received_sigterm = True