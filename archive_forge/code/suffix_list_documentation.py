from __future__ import annotations
import logging
import pkgutil
import re
from collections.abc import Sequence
from typing import cast
import requests
from requests_file import FileAdapter  # type: ignore[import-untyped]
from .cache import DiskCache
Fetch, parse, and cache the suffix lists.