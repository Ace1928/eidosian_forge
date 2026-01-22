from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def _post_batch_ingest_runs(self, body: bytes):

    def handle_429(response: requests.Response, attempt: int) -> bool:
        if response.status_code == 429:
            try:
                retry_after = float(response.headers.get('retry-after', '30'))
            except ValueError:
                logger.warning('Invalid retry-after header value: %s', response.headers.get('retry-after'))
                retry_after = 30
            retry_after = retry_after * 2 ** (attempt - 1) + random.random()
            time.sleep(retry_after)
            return True
        return False
    try:
        for api_url, api_key in self._write_api_urls.items():
            self.request_with_retries('POST', f'{api_url}/runs/batch', request_kwargs={'data': body, 'headers': {**self._headers, X_API_KEY: api_key}}, to_ignore=(ls_utils.LangSmithConflictError,), stop_after_attempt=3, handle_response=handle_429)
    except Exception as e:
        logger.warning(f'Failed to batch ingest runs: {repr(e)}')