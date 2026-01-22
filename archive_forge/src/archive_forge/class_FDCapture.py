import os
import sys
import py
import tempfile
class FDCapture:
    """ Capture IO to/from a given os-level filedescriptor. """

    def __init__(self, targetfd, tmpfile=None, now=True, patchsys=False):
        """ save targetfd descriptor, and open a new
            temporary file there.  If no tmpfile is
            specified a tempfile.Tempfile() will be opened
            in text mode.
        """
        self.targetfd = targetfd
        if tmpfile is None and targetfd != 0:
            f = tempfile.TemporaryFile('wb+')
            tmpfile = dupfile(f, encoding='UTF-8')
            f.close()
        self.tmpfile = tmpfile
        self._savefd = os.dup(self.targetfd)
        if patchsys:
            self._oldsys = getattr(sys, patchsysdict[targetfd])
        if now:
            self.start()

    def start(self):
        try:
            os.fstat(self._savefd)
        except OSError:
            raise ValueError('saved filedescriptor not valid, did you call start() twice?')
        if self.targetfd == 0 and (not self.tmpfile):
            fd = os.open(devnullpath, os.O_RDONLY)
            os.dup2(fd, 0)
            os.close(fd)
            if hasattr(self, '_oldsys'):
                setattr(sys, patchsysdict[self.targetfd], DontReadFromInput())
        else:
            os.dup2(self.tmpfile.fileno(), self.targetfd)
            if hasattr(self, '_oldsys'):
                setattr(sys, patchsysdict[self.targetfd], self.tmpfile)

    def done(self):
        """ unpatch and clean up, returns the self.tmpfile (file object)
        """
        os.dup2(self._savefd, self.targetfd)
        os.close(self._savefd)
        if self.targetfd != 0:
            self.tmpfile.seek(0)
        if hasattr(self, '_oldsys'):
            setattr(sys, patchsysdict[self.targetfd], self._oldsys)
        return self.tmpfile

    def writeorg(self, data):
        """ write a string to the original file descriptor
        """
        tempfp = tempfile.TemporaryFile()
        try:
            os.dup2(self._savefd, tempfp.fileno())
            tempfp.write(data)
        finally:
            tempfp.close()