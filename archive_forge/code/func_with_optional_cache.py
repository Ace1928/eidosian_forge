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
@contextlib.contextmanager
def with_optional_cache(path: Optional[Union[str, pathlib.Path]], ignore_hosts: Optional[Sequence[str]]=None) -> Generator[None, None, None]:
    """Use a cache for requests."""
    if path is not None:
        with with_cache(path, ignore_hosts):
            yield
    else:
        yield