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
def _set_cloud_config(self, key: str, value: t.Union[str, int, bool]) -> None:
    """Set the specified key and value in the internal configuration."""
    self.args.metadata.cloud_config[self.platform][key] = value