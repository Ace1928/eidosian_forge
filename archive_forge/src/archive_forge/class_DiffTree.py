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
class DiffTree:
    """Provides textual representations of the difference between two trees.

    A DiffTree examines two trees and where a file-id has altered
    between them, generates a textual representation of the difference.
    DiffTree uses a sequence of DiffPath objects which are each
    given the opportunity to handle a given altered fileid. The list
    of DiffPath objects can be extended globally by appending to
    DiffTree.diff_factories, or for a specific diff operation by
    supplying the extra_factories option to the appropriate method.
    """
    diff_factories = [DiffSymlink.from_diff_tree, DiffDirectory.from_diff_tree, DiffTreeReference.from_diff_tree]

    def __init__(self, old_tree, new_tree, to_file, path_encoding='utf-8', diff_text=None, extra_factories=None):
        """Constructor

        :param old_tree: Tree to show as old in the comparison
        :param new_tree: Tree to show as new in the comparison
        :param to_file: File to write comparision to
        :param path_encoding: Character encoding to write paths in
        :param diff_text: DiffPath-type object to use as a last resort for
            diffing text files.
        :param extra_factories: Factories of DiffPaths to try before any other
            DiffPaths"""
        if diff_text is None:
            diff_text = DiffText(old_tree, new_tree, to_file, path_encoding, '', '', internal_diff)
        self.old_tree = old_tree
        self.new_tree = new_tree
        self.to_file = to_file
        self.path_encoding = path_encoding
        self.differs = []
        if extra_factories is not None:
            self.differs.extend((f(self) for f in extra_factories))
        self.differs.extend((f(self) for f in self.diff_factories))
        self.differs.extend([diff_text, DiffKindChange.from_diff_tree(self)])

    @classmethod
    def from_trees_options(klass, old_tree, new_tree, to_file, path_encoding, external_diff_options, old_label, new_label, using, context_lines):
        """Factory for producing a DiffTree.

        Designed to accept options used by show_diff_trees.

        :param old_tree: The tree to show as old in the comparison
        :param new_tree: The tree to show as new in the comparison
        :param to_file: File to write comparisons to
        :param path_encoding: Character encoding to use for writing paths
        :param external_diff_options: If supplied, use the installed diff
            binary to perform file comparison, using supplied options.
        :param old_label: Prefix to use for old file labels
        :param new_label: Prefix to use for new file labels
        :param using: Commandline to use to invoke an external diff tool
        """
        if using is not None:
            extra_factories = [DiffFromTool.make_from_diff_tree(using, external_diff_options)]
        else:
            extra_factories = []
        if external_diff_options:
            opts = external_diff_options.split()

            def diff_file(olab, olines, nlab, nlines, to_file, path_encoding=None, context_lines=None):
                """:param path_encoding: not used but required
                        to match the signature of internal_diff.
                """
                external_diff(olab, olines, nlab, nlines, to_file, opts)
        else:
            diff_file = internal_diff
        diff_text = DiffText(old_tree, new_tree, to_file, path_encoding, old_label, new_label, diff_file, context_lines=context_lines)
        return klass(old_tree, new_tree, to_file, path_encoding, diff_text, extra_factories)

    def show_diff(self, specific_files, extra_trees=None):
        """Write tree diff to self.to_file

        :param specific_files: the specific files to compare (recursive)
        :param extra_trees: extra trees to use for mapping paths to file_ids
        """
        try:
            return self._show_diff(specific_files, extra_trees)
        finally:
            for differ in self.differs:
                differ.finish()

    def _show_diff(self, specific_files, extra_trees):
        iterator = self.new_tree.iter_changes(self.old_tree, specific_files=specific_files, extra_trees=extra_trees, require_versioned=True)
        has_changes = 0

        def changes_key(change):
            old_path, new_path = change.path
            path = new_path
            if path is None:
                path = old_path
            return path

        def get_encoded_path(path):
            if path is not None:
                return path.encode(self.path_encoding, 'replace')
        for change in sorted(iterator, key=changes_key):
            if not change.path[0] and (not change.path[1]) or change.kind == (None, None):
                continue
            if change.kind[0] == 'symlink' and (not self.new_tree.supports_symlinks()):
                warning('Ignoring "%s" as symlinks are not supported on this filesystem.' % (change.path[0],))
                continue
            oldpath, newpath = change.path
            oldpath_encoded = get_encoded_path(oldpath)
            newpath_encoded = get_encoded_path(newpath)
            old_present = change.kind[0] is not None and change.versioned[0]
            new_present = change.kind[1] is not None and change.versioned[1]
            executable = change.executable
            kind = change.kind
            renamed = change.renamed
            properties_changed = []
            properties_changed.extend(get_executable_change(executable[0], executable[1]))
            if properties_changed:
                prop_str = b' (properties changed: %s)' % (b', '.join(properties_changed),)
            else:
                prop_str = b''
            if (old_present, new_present) == (True, False):
                self.to_file.write(b"=== removed %s '%s'\n" % (kind[0].encode('ascii'), oldpath_encoded))
            elif (old_present, new_present) == (False, True):
                self.to_file.write(b"=== added %s '%s'\n" % (kind[1].encode('ascii'), newpath_encoded))
            elif renamed:
                self.to_file.write(b"=== renamed %s '%s' => '%s'%s\n" % (kind[0].encode('ascii'), oldpath_encoded, newpath_encoded, prop_str))
            else:
                self.to_file.write(b"=== modified %s '%s'%s\n" % (kind[0].encode('ascii'), newpath_encoded, prop_str))
            if change.changed_content:
                self._diff(oldpath, newpath, kind[0], kind[1])
                has_changes = 1
            if renamed:
                has_changes = 1
        return has_changes

    def diff(self, old_path, new_path):
        """Perform a diff of a single file

        :param old_path: The path of the file in the old tree
        :param new_path: The path of the file in the new tree
        """
        if old_path is None:
            old_kind = None
        else:
            old_kind = self.old_tree.kind(old_path)
        if new_path is None:
            new_kind = None
        else:
            new_kind = self.new_tree.kind(new_path)
        self._diff(old_path, new_path, old_kind, new_kind)

    def _diff(self, old_path, new_path, old_kind, new_kind):
        result = DiffPath._diff_many(self.differs, old_path, new_path, old_kind, new_kind)
        if result is DiffPath.CANNOT_DIFF:
            error_path = new_path
            if error_path is None:
                error_path = old_path
            raise errors.NoDiffFound(error_path)