from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_get_bytes(self, relpath):
    return self.to_transport.get_bytes(urlutils.escape(relpath))