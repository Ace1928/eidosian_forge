import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def _save_marks(self):
    if self.export_marks_file:
        revision_ids = {m: r for r, m in self.revid_to_mark.items()}
        marks_file.export_marks(self.export_marks_file, revision_ids)