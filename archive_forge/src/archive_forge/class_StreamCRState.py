import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
class StreamCRState:
    """Stream state that tracks carriage returns.

    There are two streams: stdout and stderr. We create two instances for each stream.
    An instance holds state about:
        found_cr:       if a carriage return has been found in this stream.
        cr:             most recent offset (line number) where we found \\r.
                        We update this offset with every progress bar update.
        last_normal:    most recent offset without a \\r in this stream.
                        i.e. the most recent "normal" line.
    """
    found_cr: bool
    cr: Optional[int]
    last_normal: Optional[int]

    def __init__(self) -> None:
        self.found_cr = False
        self.cr = None
        self.last_normal = None