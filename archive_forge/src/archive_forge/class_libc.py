from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
class libc:

    def inotify_add_watch(self, fd, path, mask):
        return -1