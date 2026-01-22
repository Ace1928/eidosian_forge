import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def _maybe_verify(self, repository):
    if self.patch is not None:
        if self._verify_patch(repository):
            return 'verified'
        else:
            return 'failed'
    else:
        return 'inapplicable'