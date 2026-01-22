from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _up_mkdir(self, relpath, mode):
    return self.to_transport.mkdir(urlutils.escape(relpath), mode)