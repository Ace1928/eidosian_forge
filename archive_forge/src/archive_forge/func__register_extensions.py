import os
import time
import warnings
from typing import Iterator, cast
from .. import errors, pyutils, registry, trace
def _register_extensions(self, name, extensions):
    for ext in extensions:
        self._extension_map[ext] = name