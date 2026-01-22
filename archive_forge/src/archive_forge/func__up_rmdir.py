from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_rmdir(self, relpath):
    return self.to_transport.rmdir(urlutils.escape(relpath))