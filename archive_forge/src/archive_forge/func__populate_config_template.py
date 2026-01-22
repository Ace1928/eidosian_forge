from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
@staticmethod
def _populate_config_template(template: str, values: dict[str, str]) -> str:
    """Populate and return the given template with the provided values."""
    for key in sorted(values):
        value = values[key]
        template = template.replace('@%s' % key, value)
    return template