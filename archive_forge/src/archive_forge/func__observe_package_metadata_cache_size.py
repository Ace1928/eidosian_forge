import asyncio
import http.client
import io
import json
import math
import re
import sys
import tempfile
import xmlrpc.client
from datetime import datetime, timedelta, timezone
from functools import partial
from itertools import groupby
from os import environ
from pathlib import Path
from subprocess import CalledProcessError, run
from tarfile import TarFile
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
import httpx
import tornado
from async_lru import alru_cache
from traitlets import CFloat, CInt, Unicode, config, observe
from jupyterlab._version import __version__
from jupyterlab.extensions.manager import (
@observe('package_metadata_cache_size')
def _observe_package_metadata_cache_size(self, change):
    self._fetch_package_metadata = alru_cache(maxsize=change['new'])(partial(_fetch_package_metadata, self._httpx_client))