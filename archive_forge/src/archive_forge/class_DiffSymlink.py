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
class DiffSymlink(DiffPath):

    def diff(self, old_path, new_path, old_kind, new_kind):
        """Perform comparison between two symlinks

        :param old_path: Path of the file in the old tree
        :param new_path: Path of the file in the new tree
        :param old_kind: Old file-kind of the file
        :param new_kind: New file-kind of the file
        """
        if 'symlink' not in (old_kind, new_kind):
            return self.CANNOT_DIFF
        if old_kind == 'symlink':
            old_target = self.old_tree.get_symlink_target(old_path)
        elif old_kind is None:
            old_target = None
        else:
            return self.CANNOT_DIFF
        if new_kind == 'symlink':
            new_target = self.new_tree.get_symlink_target(new_path)
        elif new_kind is None:
            new_target = None
        else:
            return self.CANNOT_DIFF
        return self.diff_symlink(old_target, new_target)

    def diff_symlink(self, old_target, new_target):
        if old_target is None:
            self.to_file.write(b"=== target is '%s'\n" % new_target.encode(self.path_encoding, 'replace'))
        elif new_target is None:
            self.to_file.write(b"=== target was '%s'\n" % old_target.encode(self.path_encoding, 'replace'))
        else:
            self.to_file.write(b"=== target changed '%s' => '%s'\n" % (old_target.encode(self.path_encoding, 'replace'), new_target.encode(self.path_encoding, 'replace')))
        return self.CHANGED