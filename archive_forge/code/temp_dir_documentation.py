import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
from pip._internal.utils.misc import enum, rmtree
Log a warning for a `rmtree` error and continue