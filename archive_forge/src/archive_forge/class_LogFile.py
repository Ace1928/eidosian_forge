import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
class LogFile(BaseLogFile):
    """
    A log file that can be rotated.

    A rotateLength of None disables automatic log rotation.
    """

    def __init__(self, name, directory, rotateLength=1000000, defaultMode=None, maxRotatedFiles=None):
        """
        Create a log file rotating on length.

        @param name: file name.
        @type name: C{str}
        @param directory: path of the log file.
        @type directory: C{str}
        @param rotateLength: size of the log file where it rotates. Default to
            1M.
        @type rotateLength: C{int}
        @param defaultMode: mode used to create the file.
        @type defaultMode: C{int}
        @param maxRotatedFiles: if not None, max number of log files the class
            creates. Warning: it removes all log files above this number.
        @type maxRotatedFiles: C{int}
        """
        BaseLogFile.__init__(self, name, directory, defaultMode)
        self.rotateLength = rotateLength
        self.maxRotatedFiles = maxRotatedFiles

    def _openFile(self):
        BaseLogFile._openFile(self)
        self.size = self._file.tell()

    def shouldRotate(self):
        """
        Rotate when the log file size is larger than rotateLength.
        """
        return self.rotateLength and self.size >= self.rotateLength

    def getLog(self, identifier):
        """
        Given an integer, return a LogReader for an old log file.
        """
        filename = '%s.%d' % (self.path, identifier)
        if not os.path.exists(filename):
            raise ValueError('no such logfile exists')
        return LogReader(filename)

    def write(self, data):
        """
        Write some data to the file.
        """
        BaseLogFile.write(self, data)
        self.size += len(data)

    def rotate(self):
        """
        Rotate the file and create a new one.

        If it's not possible to open new logfile, this will fail silently,
        and continue logging to old logfile.
        """
        if not (os.access(self.directory, os.W_OK) and os.access(self.path, os.W_OK)):
            return
        logs = self.listLogs()
        logs.reverse()
        for i in logs:
            if self.maxRotatedFiles is not None and i >= self.maxRotatedFiles:
                os.remove('%s.%d' % (self.path, i))
            else:
                os.rename('%s.%d' % (self.path, i), '%s.%d' % (self.path, i + 1))
        self._file.close()
        os.rename(self.path, '%s.1' % self.path)
        self._openFile()

    def listLogs(self):
        """
        Return sorted list of integers - the old logs' identifiers.
        """
        result = []
        for name in glob.glob('%s.*' % self.path):
            try:
                counter = int(name.split('.')[-1])
                if counter:
                    result.append(counter)
            except ValueError:
                pass
        result.sort()
        return result

    def __getstate__(self):
        state = BaseLogFile.__getstate__(self)
        del state['size']
        return state