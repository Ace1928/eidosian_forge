import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
class RestrictedGlobDirectoryWalker(GlobDirectoryWalker):
    """An restricted directory tree iterator."""

    def __init__(self, directory, pattern='*', ignore=None):
        GlobDirectoryWalker.__init__(self, directory, pattern)
        if ignore == None:
            ignore = []
        ip = [].append
        if isinstance(ignore, (tuple, list)):
            for p in ignore:
                ip(p)
        elif isinstance(ignore, str):
            ip(ignore)
        self.ignorePatterns = [_.replace('/', os.sep) for _ in ip.__self__] if os.sep != '/' else ip.__self__

    def filterFiles(self, folder, files):
        """Filters all items from files matching patterns to ignore."""
        fnm = fnmatch.fnmatch
        indicesToDelete = []
        for i, f in enumerate(files):
            for p in self.ignorePatterns:
                if fnm(f, p) or fnm(os.path.join(folder, f), p):
                    indicesToDelete.append(i)
        indicesToDelete.reverse()
        for i in indicesToDelete:
            del files[i]
        return files