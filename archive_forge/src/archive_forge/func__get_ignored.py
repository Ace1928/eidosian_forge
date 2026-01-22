from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def _get_ignored(self):
    if self._ignored is None:
        try:
            ignore_file_path = '.bzrignore-upload'
            ignore_file = self.tree.get_file(ignore_file_path)
        except transport.NoSuchFile:
            ignored_patterns = []
        else:
            ignored_patterns = ignores.parse_ignore_file(ignore_file)
        self._ignored = globbing.Globster(ignored_patterns)
    return self._ignored