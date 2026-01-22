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
def _set_retcode(self, idx: int, val: int) -> None:
    """Set retcode at process[idx] to val."""
    self._retcodes[idx] = val