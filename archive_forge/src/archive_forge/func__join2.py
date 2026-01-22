from __future__ import annotations
import binascii
import collections
import concurrent.futures
import contextlib
import hashlib
import io
import itertools
import math
import multiprocessing as mp
import os
import re
import shutil
import stat as stat_module
import tempfile
import time
import urllib.parse
from functools import partial
from types import ModuleType
from typing import (
import urllib3
import filelock
from blobfile import _azure as azure
from blobfile import _common as common
from blobfile import _gcp as gcp
from blobfile._common import (
def _join2(conf: Config, a: str, b: str) -> str:
    if _is_local_path(a):
        return os.path.join(a, b)
    elif _is_gcp_path(a):
        return gcp.join_paths(conf, a, b)
    elif _is_azure_path(a):
        return azure.join_paths(conf, a, b)
    else:
        raise Error(f"Unrecognized path: '{a}'")