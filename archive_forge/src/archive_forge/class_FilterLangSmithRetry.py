import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
class FilterLangSmithRetry(logging.Filter):
    """Filter for retries from this lib."""

    def filter(self, record) -> bool:
        """Filter retries from this library."""
        msg = record.getMessage()
        return 'LangSmithRetry' not in msg