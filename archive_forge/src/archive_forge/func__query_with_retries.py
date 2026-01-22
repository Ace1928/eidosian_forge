from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
import openai
from typing_extensions import override
def _query_with_retries(self, func: Callable[..., str], *args: str, retries: int=NUM_LLM_RETRIES, backoff_factor: float=0.5) -> str:
    last_exception = None
    for retry in range(retries):
        try:
            return func(*args)
        except Exception as exception:
            last_exception = exception
            sleep_time = backoff_factor * 2 ** retry
            time.sleep(sleep_time)
            LOG.debug(f'LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds...')
    raise RuntimeError(f'Unable to query LLM after {retries} retries: {last_exception}')