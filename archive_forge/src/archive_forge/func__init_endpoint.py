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
def _init_endpoint(self) -> None:
    settings = self._api.settings()
    settings.update(self._settings)
    self._endpoint = '{base}/files/{entity}/{project}/{run}/file_stream'.format(base=settings['base_url'], entity=settings['entity'], project=settings['project'], run=self._run_id)