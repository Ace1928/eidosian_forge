import os
import re
import shutil
import tempfile
from datetime import datetime
from time import time
from typing import List, Optional
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method
from cmdstanpy.utils import get_logger
def _set_timeout_flag(self, idx: int, val: bool) -> None:
    """Set timeout_flag at process[idx] to val."""
    self._timeout_flags[idx] = val