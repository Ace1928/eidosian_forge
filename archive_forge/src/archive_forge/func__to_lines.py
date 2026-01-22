import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def _to_lines(self, base_revision=False):
    """Serialize as a list of lines

        :return: a list of lines
        """
    time_str = timestamp.format_patch_date(self.time, self.timezone)
    stanza = rio.Stanza(revision_id=self.revision_id, timestamp=time_str, target_branch=self.target_branch, testament_sha1=self.testament_sha1)
    for key in ('source_branch', 'message'):
        if self.__dict__[key] is not None:
            stanza.add(key, self.__dict__[key])
    if base_revision:
        stanza.add('base_revision_id', self.base_revision_id)
    lines = [b'# ' + self._format_string + b'\n']
    lines.extend(rio.to_patch_lines(stanza))
    lines.append(b'# \n')
    return lines