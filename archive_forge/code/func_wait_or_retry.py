import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
def wait_or_retry(self, max_retries: int=2, backoff_s: int=5):
    assert max_retries > 0
    last_error_traceback = None
    for i in range(max_retries + 1):
        try:
            self.wait()
        except Exception as e:
            attempts_remaining = max_retries - i
            if attempts_remaining == 0:
                last_error_traceback = traceback.format_exc()
                break
            logger.error(f'The latest sync operation failed with the following error: {repr(e)}\nRetrying {attempts_remaining} more time(s) after sleeping for {backoff_s} seconds...')
            time.sleep(backoff_s)
            self.retry()
            continue
        return
    raise RuntimeError(f'Failed sync even after {max_retries} retries. The latest sync failed with the following error:\n{last_error_traceback}')