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
class CloudEnvironmentConfig:
    """Configuration for the environment."""

    def __init__(self, env_vars: t.Optional[dict[str, str]]=None, ansible_vars: t.Optional[dict[str, t.Any]]=None, module_defaults: t.Optional[dict[str, dict[str, t.Any]]]=None, callback_plugins: t.Optional[list[str]]=None):
        self.env_vars = env_vars
        self.ansible_vars = ansible_vars
        self.module_defaults = module_defaults
        self.callback_plugins = callback_plugins