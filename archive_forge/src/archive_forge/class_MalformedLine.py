import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
class MalformedLine(PatchSyntax):
    _fmt = 'Malformed line.  %(desc)s\n%(line)r'

    def __init__(self, desc, line):
        self.desc = desc
        self.line = line