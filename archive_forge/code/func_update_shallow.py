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
def update_shallow(self, new_shallow, new_unshallow):
    """Update the list of shallow objects.

        Args:
          new_shallow: Newly shallow objects
          new_unshallow: Newly no longer shallow objects
        """
    shallow = self.get_shallow()
    if new_shallow:
        shallow.update(new_shallow)
    if new_unshallow:
        shallow.difference_update(new_unshallow)
    if shallow:
        self._put_named_file('shallow', b''.join([sha + b'\n' for sha in shallow]))
    else:
        self._del_named_file('shallow')