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
def _get_cloud_config(self, key: str, default: t.Optional[t.Union[str, int, bool]]=None) -> t.Union[str, int, bool]:
    """Return the specified value from the internal configuration."""
    if default is not None:
        return self.args.metadata.cloud_config[self.platform].get(key, default)
    return self.args.metadata.cloud_config[self.platform][key]