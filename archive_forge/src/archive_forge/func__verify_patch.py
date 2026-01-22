import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def _verify_patch(self, repository):
    calculated_patch = self._generate_diff(repository, self.revision_id, self.base_revision_id)
    stored_patch = re.sub(b'\r\n?', b'\n', self.patch)
    calculated_patch = re.sub(b'\r\n?', b'\n', calculated_patch)
    calculated_patch = re.sub(b' *\n', b'\n', calculated_patch)
    stored_patch = re.sub(b' *\n', b'\n', stored_patch)
    return calculated_patch == stored_patch