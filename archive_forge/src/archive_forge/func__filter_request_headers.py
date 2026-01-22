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
def _filter_request_headers(request: Any) -> Any:
    if ignore_hosts and any((request.url.startswith(host) for host in ignore_hosts)):
        return None
    request.headers = {}
    return request