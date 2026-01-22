import os
import re
import shutil
import sys
class ReadFileFilter(CommandFilter):
    """Specific filter for the utils.read_file_as_root call."""

    def __init__(self, file_path, *args):
        self.file_path = file_path
        super(ReadFileFilter, self).__init__('/bin/cat', 'root', *args)

    def match(self, userargs):
        return userargs == ['cat', self.file_path]