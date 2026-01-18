from __future__ import annotations
import asyncio
import time
import functools
from typing import (
from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry.retry_base import exponential_sleep_generator
from google.api_core.retry.retry_base import build_retry_error
from google.api_core.retry.retry_base import RetryFailureReason
from google.api_core.retry.retry_base import if_exception_type  # noqa
from google.api_core.retry.retry_base import if_transient_error  # noqa
A wrapper that calls target function with retry.