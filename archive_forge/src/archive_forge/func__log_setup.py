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
def _log_setup(self, settings: Settings) -> None:
    """Set up logging from settings."""
    filesystem.mkdir_exists_ok(os.path.dirname(settings.log_user))
    filesystem.mkdir_exists_ok(os.path.dirname(settings.log_internal))
    filesystem.mkdir_exists_ok(os.path.dirname(settings.sync_file))
    filesystem.mkdir_exists_ok(settings.files_dir)
    filesystem.mkdir_exists_ok(settings._tmp_code_dir)
    if settings.symlink:
        self._safe_symlink(os.path.dirname(settings.sync_symlink_latest), os.path.dirname(settings.sync_file), os.path.basename(settings.sync_symlink_latest), delete=True)
        self._safe_symlink(os.path.dirname(settings.log_symlink_user), settings.log_user, os.path.basename(settings.log_symlink_user), delete=True)
        self._safe_symlink(os.path.dirname(settings.log_symlink_internal), settings.log_internal, os.path.basename(settings.log_symlink_internal), delete=True)
    _set_logger(logging.getLogger('wandb'))
    self._enable_logging(settings.log_user)
    assert self._wl
    assert logger
    self._wl._early_logger_flush(logger)
    logger.info(f'Logging user logs to {settings.log_user}')
    logger.info(f'Logging internal logs to {settings.log_internal}')