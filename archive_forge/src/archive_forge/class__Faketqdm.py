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
class _Faketqdm:

    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        self.total = total
        self.disable = disable
        self.n = 0

    def update(self, n):
        if self.disable:
            return
        self.n += n
        if self.total is None:
            sys.stderr.write(f'\r{self.n:.1f} bytes')
        else:
            sys.stderr.write(f'\r{100 * self.n / float(self.total):.1f}%')
        sys.stderr.flush()

    def set_description(self, *args, **kwargs):
        pass

    def write(self, s):
        sys.stderr.write(f'{s}\n')

    def close(self):
        self.disable = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return
        sys.stderr.write('\n')