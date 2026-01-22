import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def _read_heads(self, name):
    f = self.get_named_file(name)
    if f is None:
        return []
    with f:
        return [line.strip() for line in f.readlines() if line.strip()]