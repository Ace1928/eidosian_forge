import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
def _egg_fragment(self) -> Optional[str]:
    match = self._egg_fragment_re.search(self._url)
    if not match:
        return None
    project_name = match.group(1)
    if not self._project_name_re.match(project_name):
        deprecated(reason=f'{self} contains an egg fragment with a non-PEP 508 name', replacement='to use the req @ url syntax, and remove the egg fragment', gone_in='25.0', issue=11617)
    return project_name