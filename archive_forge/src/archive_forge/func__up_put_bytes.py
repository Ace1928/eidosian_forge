from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_put_bytes(self, relpath, bytes, mode):
    self.to_transport.put_bytes(urlutils.escape(relpath), bytes, mode)