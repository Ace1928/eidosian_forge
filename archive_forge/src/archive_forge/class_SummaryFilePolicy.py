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
class SummaryFilePolicy(DefaultFilePolicy):

    def process_chunks(self, chunks: List[Chunk]) -> Union[bool, 'ProcessedChunk']:
        data = chunks[-1].data
        if len(data) > util.MAX_LINE_BYTES:
            msg = 'Summary data exceeds maximum size of {}. Dropping it.'.format(util.to_human_size(util.MAX_LINE_BYTES))
            wandb.termerror(msg, repeat=False)
            wandb._sentry.message(msg, repeat=False)
            return False
        return {'offset': 0, 'content': [data]}