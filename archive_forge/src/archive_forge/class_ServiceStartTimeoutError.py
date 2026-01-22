import datetime
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, Optional
from wandb import _sentry, termlog
from wandb.env import error_reporting_enabled
from wandb.errors import Error
from wandb.sdk.lib.wburls import wburls
from wandb.util import get_core_path, get_module
from . import _startup_debug, port_file
from .service_base import ServiceInterface
from .service_sock import ServiceSockInterface
class ServiceStartTimeoutError(Error):
    """Raised when service start times out."""
    pass