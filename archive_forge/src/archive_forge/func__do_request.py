import dataclasses
import importlib
import logging
import json
import os
import yaml
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union
from pkg_resources import packaging
import ray
import ssl
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.dashboard.modules.job.common import uri_to_http_components
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.utils import split_address
from ray.autoscaler._private.cli_logger import cli_logger
def _do_request(self, method: str, endpoint: str, *, data: Optional[bytes]=None, json_data: Optional[dict]=None, **kwargs) -> 'requests.Response':
    """Perform the actual HTTP request

        Keyword arguments other than "cookies", "headers" are forwarded to the
        `requests.request()`.
        """
    url = self._address + endpoint
    logger.debug(f'Sending request to {url} with json data: {json_data or {}}.')
    return requests.request(method, url, cookies=self._cookies, data=data, json=json_data, headers=self._headers, verify=self._verify, **kwargs)