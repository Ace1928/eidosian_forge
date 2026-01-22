from __future__ import division
import re
import stat
from .helpers import (
class FileCopyCommand(FileCommand):

    def __init__(self, src_path, dest_path):
        FileCommand.__init__(self, b'filecopy')
        self.src_path = check_path(src_path)
        self.dest_path = check_path(dest_path)

    def __bytes__(self):
        return b' '.join([b'C', format_path(self.src_path, quote_spaces=True), format_path(self.dest_path)])