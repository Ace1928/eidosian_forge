from __future__ import annotations
from typing import (
import asyncio
import time
import sys
import functools
from google.api_core.retry.retry_base import _BaseRetry
from google.api_core.retry.retry_base import _retry_error_helper
from google.api_core.retry import exponential_sleep_generator
from google.api_core.retry import build_retry_error
from google.api_core.retry import RetryFailureReason
A wrapper that calls target function with retry.