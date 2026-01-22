import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class RemoveLine(HunkLine):

    def __init__(self, contents):
        HunkLine.__init__(self, contents)

    def as_bytes(self):
        return self.get_str(b'-')