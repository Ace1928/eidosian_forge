import functools
import logging
import logging.config
import optparse
import os
import sys
import traceback
from optparse import Values
from typing import Any, Callable, List, Optional, Tuple
from pip._vendor.rich import traceback as rich_traceback
from pip._internal.cli import cmdoptions
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.cli.parser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip._internal.cli.status_codes import (
from pip._internal.exceptions import (
from pip._internal.utils.filesystem import check_path_owner
from pip._internal.utils.logging import BrokenStdoutLoggingError, setup_logging
from pip._internal.utils.misc import get_prog, normalize_path
from pip._internal.utils.temp_dir import TempDirectoryTypeRegistry as TempDirRegistry
from pip._internal.utils.temp_dir import global_tempdir_manager, tempdir_registry
from pip._internal.utils.virtualenv import running_under_virtualenv

        This is a no-op so that commands by default do not do the pip version
        check.
        