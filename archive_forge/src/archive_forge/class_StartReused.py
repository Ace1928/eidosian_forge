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
class StartReused:
    """Possible return value of the `start` function.

    Indicates that a call to `start` was compatible with an existing
    TensorBoard process, which can be reused according to the provided
    info.

    Attributes:
      info: A `TensorBoardInfo` object.
    """
    info: TensorBoardInfo