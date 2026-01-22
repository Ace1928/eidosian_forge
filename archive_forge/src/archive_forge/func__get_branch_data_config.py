import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def _get_branch_data_config(self):
    if self._branch_data_config is None:
        self._branch_data_config = TreeConfig(self.branch)
        self._branch_data_config.config_id = self.config_id
    return self._branch_data_config