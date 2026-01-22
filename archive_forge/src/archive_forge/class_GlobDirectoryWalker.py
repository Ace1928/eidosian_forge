import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
class GlobDirectoryWalker:
    """A forward iterator that traverses files in a directory tree."""

    def __init__(self, directory, pattern='*'):
        self.index = 0
        self.pattern = pattern
        directory.replace('/', os.sep)
        if os.path.isdir(directory):
            self.stack = [directory]
            self.files = []
        else:
            if not isCompactDistro() or not __rl_loader__ or (not rl_isdir(directory)):
                raise ValueError('"%s" is not a directory' % directory)
            self.directory = directory[len(__rl_loader__.archive) + len(os.sep):]
            pfx = self.directory + os.sep
            n = len(pfx)
            self.files = list(map(lambda x, n=n: x[n:], list(filter(lambda x, pfx=pfx: x.startswith(pfx), list(__rl_loader__._files.keys())))))
            self.files.sort()
            self.stack = []

    def __getitem__(self, index):
        while 1:
            try:
                file = self.files[self.index]
                self.index = self.index + 1
            except IndexError:
                self.directory = self.stack.pop()
                self.files = os.listdir(self.directory)
                self.files = self.filterFiles(self.directory, self.files)
                self.index = 0
            else:
                fullname = os.path.join(self.directory, file)
                if os.path.isdir(fullname) and (not os.path.islink(fullname)):
                    self.stack.append(fullname)
                if fnmatch.fnmatch(file, self.pattern):
                    return fullname

    def filterFiles(self, folder, files):
        """Filter hook, overwrite in subclasses as needed."""
        return files