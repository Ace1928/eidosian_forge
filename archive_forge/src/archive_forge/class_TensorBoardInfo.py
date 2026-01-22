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
class TensorBoardInfo:
    """Holds the information about a running TensorBoard instance.

    Attributes:
      version: Version of the running TensorBoard.
      start_time: Seconds since epoch.
      pid: ID of the process running TensorBoard.
      port: Port on which TensorBoard is running.
      path_prefix: Relative prefix to the path, may be empty.
      logdir: Data location used by the TensorBoard server, may be empty.
      db: Database connection used by the TensorBoard server, may be empty.
      cache_key: Opaque, as given by `cache_key` below.
    """
    version: str
    start_time: int
    pid: int
    port: int
    path_prefix: str
    logdir: str
    db: str
    cache_key: str