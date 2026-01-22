import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
@default('nodejs')
def _default_nodejs(self):
    return shutil.which('node') or shutil.which('nodejs') or shutil.which('nodejs.exe')