import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
class _PrematchedMatcher(difflib.SequenceMatcher):
    """Allow SequenceMatcher operations to use predetermined blocks"""

    def __init__(self, matching_blocks):
        difflib.SequenceMatcher(self, None, None)
        self.matching_blocks = matching_blocks
        self.opcodes = None