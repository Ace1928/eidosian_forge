import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
class DirectorySandbox(AbstractSandbox):
    """Restrict operations to a single subdirectory - pseudo-chroot"""
    write_ops = dict.fromkeys(['open', 'chmod', 'chown', 'mkdir', 'remove', 'unlink', 'rmdir', 'utime', 'lchown', 'chroot', 'mkfifo', 'mknod', 'tempnam'])
    _exception_patterns = []
    'exempt writing to paths that match the pattern'

    def __init__(self, sandbox, exceptions=_EXCEPTIONS):
        self._sandbox = os.path.normcase(os.path.realpath(sandbox))
        self._prefix = os.path.join(self._sandbox, '')
        self._exceptions = [os.path.normcase(os.path.realpath(path)) for path in exceptions]
        AbstractSandbox.__init__(self)

    def _violation(self, operation, *args, **kw):
        from setuptools.sandbox import SandboxViolation
        raise SandboxViolation(operation, args, kw)
    if _file:

        def _file(self, path, mode='r', *args, **kw):
            if mode not in ('r', 'rt', 'rb', 'rU', 'U') and (not self._ok(path)):
                self._violation('file', path, mode, *args, **kw)
            return _file(path, mode, *args, **kw)

    def _open(self, path, mode='r', *args, **kw):
        if mode not in ('r', 'rt', 'rb', 'rU', 'U') and (not self._ok(path)):
            self._violation('open', path, mode, *args, **kw)
        return _open(path, mode, *args, **kw)

    def tmpnam(self):
        self._violation('tmpnam')

    def _ok(self, path):
        active = self._active
        try:
            self._active = False
            realpath = os.path.normcase(os.path.realpath(path))
            return self._exempted(realpath) or realpath == self._sandbox or realpath.startswith(self._prefix)
        finally:
            self._active = active

    def _exempted(self, filepath):
        start_matches = (filepath.startswith(exception) for exception in self._exceptions)
        pattern_matches = (re.match(pattern, filepath) for pattern in self._exception_patterns)
        candidates = itertools.chain(start_matches, pattern_matches)
        return any(candidates)

    def _remap_input(self, operation, path, *args, **kw):
        """Called for path inputs"""
        if operation in self.write_ops and (not self._ok(path)):
            self._violation(operation, os.path.realpath(path), *args, **kw)
        return path

    def _remap_pair(self, operation, src, dst, *args, **kw):
        """Called for path pairs like rename, link, and symlink operations"""
        if not self._ok(src) or not self._ok(dst):
            self._violation(operation, src, dst, *args, **kw)
        return (src, dst)

    def open(self, file, flags, mode=511, *args, **kw):
        """Called for low-level os.open()"""
        if flags & WRITE_FLAGS and (not self._ok(file)):
            self._violation('os.open', file, flags, mode, *args, **kw)
        return _os.open(file, flags, mode, *args, **kw)