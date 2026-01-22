import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def _find_http_files(self, options: Values) -> List[str]:
    old_http_dir = self._cache_dir(options, 'http')
    new_http_dir = self._cache_dir(options, 'http-v2')
    return filesystem.find_files(old_http_dir, '*') + filesystem.find_files(new_http_dir, '*')