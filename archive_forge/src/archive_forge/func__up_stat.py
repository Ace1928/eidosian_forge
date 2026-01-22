from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_stat(self, relpath):
    return self.to_transport.stat(urlutils.escape(relpath))