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
class DiffFromTool(DiffPath):

    def __init__(self, command_template: Union[str, List[str]], old_tree: Tree, new_tree: Tree, to_file, path_encoding='utf-8'):
        import tempfile
        DiffPath.__init__(self, old_tree, new_tree, to_file, path_encoding)
        self.command_template = command_template
        import tempfile
        self._root = tempfile.mkdtemp(prefix='brz-diff-')

    @classmethod
    def from_string(klass, command_template: Union[str, List[str]], old_tree: Tree, new_tree: Tree, to_file, path_encoding: str='utf-8'):
        return klass(command_template, old_tree, new_tree, to_file, path_encoding)

    @classmethod
    def make_from_diff_tree(klass, command_string, external_diff_options=None):

        def from_diff_tree(diff_tree):
            full_command_string = [command_string]
            if external_diff_options is not None:
                full_command_string.extend(external_diff_options.split())
            return klass.from_string(full_command_string, diff_tree.old_tree, diff_tree.new_tree, diff_tree.to_file)
        return from_diff_tree

    def _get_command(self, old_path, new_path):
        my_map = {'old_path': old_path, 'new_path': new_path}
        command = [t.format(**my_map) for t in self.command_template]
        if command == self.command_template:
            command += [old_path, new_path]
        if sys.platform == 'win32':
            command_encoded = []
            for c in command:
                if isinstance(c, str):
                    command_encoded.append(c.encode('mbcs'))
                else:
                    command_encoded.append(c)
            return command_encoded
        else:
            return command

    def _execute(self, old_path, new_path):
        command = self._get_command(old_path, new_path)
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=self._root)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise errors.ExecutableMissing(command[0])
            else:
                raise
        self.to_file.write(proc.stdout.read())
        proc.stdout.close()
        return proc.wait()

    def _try_symlink_root(self, tree, prefix):
        if getattr(tree, 'abspath', None) is None or not osutils.supports_symlinks(self._root):
            return False
        try:
            os.symlink(tree.abspath(''), osutils.pathjoin(self._root, prefix))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return True

    @staticmethod
    def _fenc():
        """Returns safe encoding for passing file path to diff tool"""
        if sys.platform == 'win32':
            return 'mbcs'
        else:
            return sys.getfilesystemencoding() or 'ascii'

    def _is_safepath(self, path):
        """Return true if `path` may be able to pass to subprocess."""
        fenc = self._fenc()
        try:
            return path == path.encode(fenc).decode(fenc)
        except UnicodeError:
            return False

    def _safe_filename(self, prefix, relpath):
        """Replace unsafe character in `relpath` then join `self._root`,
        `prefix` and `relpath`."""
        fenc = self._fenc()
        relpath_tmp = relpath.encode(fenc, 'replace').decode(fenc, 'replace')
        relpath_tmp = relpath_tmp.replace('?', '_')
        return osutils.pathjoin(self._root, prefix, relpath_tmp)

    def _write_file(self, relpath, tree, prefix, force_temp=False, allow_write=False):
        if not force_temp and isinstance(tree, WorkingTree):
            full_path = tree.abspath(relpath)
            if self._is_safepath(full_path):
                return full_path
        full_path = self._safe_filename(prefix, relpath)
        if not force_temp and self._try_symlink_root(tree, prefix):
            return full_path
        parent_dir = osutils.dirname(full_path)
        try:
            os.makedirs(parent_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        with tree.get_file(relpath) as source, open(full_path, 'wb') as target:
            osutils.pumpfile(source, target)
        try:
            mtime = tree.get_file_mtime(relpath)
        except FileTimestampUnavailable:
            pass
        else:
            os.utime(full_path, (mtime, mtime))
        if not allow_write:
            osutils.make_readonly(full_path)
        return full_path

    def _prepare_files(self, old_path, new_path, force_temp=False, allow_write_new=False):
        old_disk_path = self._write_file(old_path, self.old_tree, 'old', force_temp)
        new_disk_path = self._write_file(new_path, self.new_tree, 'new', force_temp, allow_write=allow_write_new)
        return (old_disk_path, new_disk_path)

    def finish(self):
        try:
            osutils.rmtree(self._root)
        except OSError as e:
            if e.errno != errno.ENOENT:
                mutter('The temporary directory "%s" was not cleanly removed: %s.' % (self._root, e))

    def diff(self, old_path, new_path, old_kind, new_kind):
        if (old_kind, new_kind) != ('file', 'file'):
            return DiffPath.CANNOT_DIFF
        old_disk_path, new_disk_path = self._prepare_files(old_path, new_path)
        self._execute(old_disk_path, new_disk_path)

    def edit_file(self, old_path, new_path):
        """Use this tool to edit a file.

        A temporary copy will be edited, and the new contents will be
        returned.

        :return: The new contents of the file.
        """
        old_abs_path, new_abs_path = self._prepare_files(old_path, new_path, allow_write_new=True, force_temp=True)
        command = self._get_command(old_abs_path, new_abs_path)
        subprocess.call(command, cwd=self._root)
        with open(new_abs_path, 'rb') as new_file:
            return new_file.read()