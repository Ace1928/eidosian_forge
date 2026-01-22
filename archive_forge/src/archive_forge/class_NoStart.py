from __future__ import annotations
import logging
import os
import sys
import typing as t
from copy import deepcopy
from pathlib import Path
from shutil import which
from traitlets import Bool, List, Unicode, observe
from traitlets.config.application import Application, catch_config_error
from traitlets.config.loader import ConfigFileNotFound
from .paths import (
from .utils import ensure_dir_exists, ensure_event_loop
class NoStart(Exception):
    """Exception to raise when an application shouldn't start"""