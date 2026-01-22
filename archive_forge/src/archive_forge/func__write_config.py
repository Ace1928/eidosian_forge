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
def _write_config(self, content: str) -> None:
    """Write the given content to the config file."""
    prefix = '%s-' % os.path.splitext(os.path.basename(self.config_static_path))[0]
    with tempfile.NamedTemporaryFile(dir=data_context().content.integration_path, prefix=prefix, suffix=self.config_extension, delete=False) as config_fd:
        filename = os.path.join(data_context().content.integration_path, os.path.basename(config_fd.name))
        self.config_path = filename
        self.remove_config = True
        display.info('>>> Config: %s\n%s' % (filename, content.strip()), verbosity=3)
        config_fd.write(to_bytes(content))
        config_fd.flush()