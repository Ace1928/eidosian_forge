from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def upload_symlink(self, relpath, target):
    self.to_transport.symlink(target, relpath)