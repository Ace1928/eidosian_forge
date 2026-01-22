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
def _handle_response(self, response: Union[Exception, 'requests.Response']) -> None:
    """Log dropped chunks and updates dynamic settings."""
    if isinstance(response, Exception):
        wandb.termerror('Dropped streaming file chunk (see wandb/debug-internal.log)')
        logger.exception('dropped chunk %s' % response)
        self._dropped_chunks += 1
    else:
        parsed: Optional[dict] = None
        try:
            parsed = response.json()
        except Exception:
            pass
        if isinstance(parsed, dict):
            limits = parsed.get('limits')
            if isinstance(limits, dict):
                self._api.dynamic_settings.update(limits)