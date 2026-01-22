import base64
import dataclasses
import datetime
import errno
import json
import os
import subprocess
import tempfile
import time
import typing
from typing import Optional
from tensorboard import version
from tensorboard.util import tb_logging
@dataclasses.dataclass(frozen=True)
class StartTimedOut:
    """Possible return value of the `start` function.

    Indicates that a call to `start` launched a TensorBoard process, but
    that process neither exited nor wrote its info file within the allowed
    timeout period. The process may still be running under the included
    PID.

    Attributes:
      pid: ID of the process running TensorBoard.
    """
    pid: int