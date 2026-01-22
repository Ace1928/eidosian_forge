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
def _jupyter_setup(self, settings: Settings) -> None:
    """Add hooks, and session history saving."""
    self.notebook = wandb.jupyter.Notebook(settings)
    ipython = self.notebook.shell
    if not hasattr(ipython.display_pub, '_orig_publish'):
        logger.info('configuring jupyter hooks %s', self)
        ipython.display_pub._orig_publish = ipython.display_pub.publish
        ipython.events.register('pre_run_cell', self._resume_backend)
        ipython.events.register('post_run_cell', self._pause_backend)
        self._teardown_hooks.append(TeardownHook(self._jupyter_teardown, TeardownStage.EARLY))

    def publish(data, metadata=None, **kwargs) -> None:
        ipython.display_pub._orig_publish(data, metadata=metadata, **kwargs)
        assert self.notebook is not None
        self.notebook.save_display(ipython.execution_count, {'data': data, 'metadata': metadata})
    ipython.display_pub.publish = publish