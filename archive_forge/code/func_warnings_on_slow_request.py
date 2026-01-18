import logging
import threading
import urllib
import warnings
from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import requests
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.utils import (
from ray.util.annotations import DeveloperAPI
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException, ServerUnavailable
@contextmanager
def warnings_on_slow_request(*, address: str, endpoint: str, timeout: float, explain: bool):
    """A context manager to print warnings if the request is replied slowly.

    Warnings are printed 3 times

    Args:
        address: The address of the endpoint.
        endpoint: The name of the endpoint.
        timeout: Request timeout in seconds.
        explain: Whether ot not it will print the warning.
    """
    if not explain:
        yield
        return

    def print_warning(elapsed: float):
        logger.info(f'({round(elapsed, 2)} / {timeout} seconds) Waiting for the response from the API server address {address}{endpoint}.')
    warning_timers = [threading.Timer(timeout / i, print_warning, args=[timeout / i]) for i in [2, 4, 8]]
    try:
        for timer in warning_timers:
            timer.start()
        yield
    finally:
        for timer in warning_timers:
            timer.cancel()