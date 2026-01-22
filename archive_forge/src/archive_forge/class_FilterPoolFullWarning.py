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
class FilterPoolFullWarning(logging.Filter):
    """Filter urrllib3 warnings logged when the connection pool isn't reused."""

    def __init__(self, name: str='', host: str='') -> None:
        """Initialize the FilterPoolFullWarning filter.

        Args:
            name (str, optional): The name of the filter. Defaults to "".
            host (str, optional): The host to filter. Defaults to "".
        """
        super().__init__(name)
        self._host = host

    def filter(self, record) -> bool:
        """urllib3.connectionpool:Connection pool is full, discarding connection: ..."""
        msg = record.getMessage()
        if 'Connection pool is full, discarding connection' not in msg:
            return True
        return self._host not in msg