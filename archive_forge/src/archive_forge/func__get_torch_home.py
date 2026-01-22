import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import torch
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
from torch.serialization import MAP_LOCATION
def _get_torch_home():
    torch_home = os.path.expanduser(os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home