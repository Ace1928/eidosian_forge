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
Normalize extension name.

        Remove `@` from npm scope and replace `/` and `_` by `-`.

        Args:
            name: Extension name
        Returns:
            Normalized name
        