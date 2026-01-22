import copy
import json
import logging
import os
import platform
import sys
import tempfile
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
import wandb
import wandb.env
from wandb import trigger
from wandb.errors import CommError, Error, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.integration import sagemaker
from wandb.integration.magic import magic_install
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import StrPath
from wandb.util import _is_artifact_representation
from . import wandb_login, wandb_setup
from .backend.backend import Backend
from .lib import (
from .lib.deprecate import Deprecated, deprecate
from .lib.mailbox import Mailbox, MailboxProgress
from .lib.printer import Printer, get_printer
from .lib.wburls import wburls
from .wandb_helper import parse_config
from .wandb_run import Run, TeardownHook, TeardownStage
from .wandb_settings import Settings, Source
def _enable_logging(self, log_fname: str, run_id: Optional[str]=None) -> None:
    """Enable logging to the global debug log.

        This adds a run_id to the log, in case of multiple processes on the same machine.
        Currently, there is no way to disable logging after it's enabled.
        """
    handler = logging.FileHandler(log_fname)
    handler.setLevel(logging.INFO)

    class WBFilter(logging.Filter):

        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = run_id
            return True
    if run_id:
        formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(threadName)-10s:%(process)d [%(run_id)s:%(filename)s:%(funcName)s():%(lineno)s] %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(threadName)-10s:%(process)d [%(filename)s:%(funcName)s():%(lineno)s] %(message)s')
    handler.setFormatter(formatter)
    if run_id:
        handler.addFilter(WBFilter())
    assert logger is not None
    logger.propagate = False
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    self._teardown_hooks.append(TeardownHook(lambda: (handler.close(), logger.removeHandler(handler)), TeardownStage.LATE))