import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class PatchConflict(BzrError):
    _fmt = 'Text contents mismatch at line %(line_no)d.  Original has "%(orig_line)s", but patch says it should be "%(patch_line)s"'

    def __init__(self, line_no, orig_line, patch_line):
        self.line_no = line_no
        self.orig_line = orig_line.rstrip('\n')
        self.patch_line = patch_line.rstrip('\n')