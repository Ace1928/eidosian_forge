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
class DiffText(DiffPath):
    EPOCH_DATE = '1970-01-01 00:00:00 +0000'

    def __init__(self, old_tree, new_tree, to_file, path_encoding='utf-8', old_label='', new_label='', text_differ=internal_diff, context_lines=DEFAULT_CONTEXT_AMOUNT):
        DiffPath.__init__(self, old_tree, new_tree, to_file, path_encoding)
        self.text_differ = text_differ
        self.old_label = old_label
        self.new_label = new_label
        self.path_encoding = path_encoding
        self.context_lines = context_lines

    def diff(self, old_path, new_path, old_kind, new_kind):
        """Compare two files in unified diff format

        :param old_path: Path of the file in the old tree
        :param new_path: Path of the file in the new tree
        :param old_kind: Old file-kind of the file
        :param new_kind: New file-kind of the file
        """
        if 'file' not in (old_kind, new_kind):
            return self.CANNOT_DIFF
        if old_kind == 'file':
            old_date = _patch_header_date(self.old_tree, old_path)
        elif old_kind is None:
            old_date = self.EPOCH_DATE
        else:
            return self.CANNOT_DIFF
        if new_kind == 'file':
            new_date = _patch_header_date(self.new_tree, new_path)
        elif new_kind is None:
            new_date = self.EPOCH_DATE
        else:
            return self.CANNOT_DIFF
        from_label = '{}{}\t{}'.format(self.old_label, old_path or new_path, old_date)
        to_label = '{}{}\t{}'.format(self.new_label, new_path or old_path, new_date)
        return self.diff_text(old_path, new_path, from_label, to_label)

    def diff_text(self, from_path, to_path, from_label, to_label):
        """Diff the content of given files in two trees

        :param from_path: The path in the from tree. If None,
            the file is not present in the from tree.
        :param to_path: The path in the to tree. This may refer
            to a different file from from_path.  If None,
            the file is not present in the to tree.
        """

        def _get_text(tree, path):
            if path is None:
                return []
            try:
                return tree.get_file_lines(path)
            except _mod_transport.NoSuchFile:
                return []
        try:
            from_text = _get_text(self.old_tree, from_path)
            to_text = _get_text(self.new_tree, to_path)
            self.text_differ(from_label, from_text, to_label, to_text, self.to_file, path_encoding=self.path_encoding, context_lines=self.context_lines)
        except errors.BinaryFile:
            self.to_file.write(('Binary files %s%s and %s%s differ\n' % (self.old_label, from_path or to_path, self.new_label, to_path or from_path)).encode(self.path_encoding, 'replace'))
        return self.CHANGED