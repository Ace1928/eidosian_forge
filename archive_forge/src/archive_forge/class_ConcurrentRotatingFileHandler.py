from builtins import range
import os
import sys
from random import randint
from logging import Handler
from logging.handlers import BaseRotatingHandler
from filelock import SoftFileLock
import logging.handlers
class ConcurrentRotatingFileHandler(BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file to the
    next when the current file reaches a certain size. Multiple processes can
    write to the log file concurrently, but this may mean that the file will
    exceed the given size.
    """

    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, debug=True, supress_abs_warn=False):
        """
        Open the specified file and use it as the stream for logging.

        By default, the file grows indefinitely. You can specify particular
        values of maxBytes and backupCount to allow the file to rollover at
        a predetermined size.

        Rollover occurs whenever the current log file is nearly maxBytes in
        length. If backupCount is >= 1, the system will successively create
        new files with the same pathname as the base file, but with extensions
        ".1", ".2" etc. appended to it. For example, with a backupCount of 5
        and a base file name of "app.log", you would get "app.log",
        "app.log.1", "app.log.2", ... through to "app.log.5". The file being
        written to is always "app.log" - when it gets filled up, it is closed
        and renamed to "app.log.1", and if files "app.log.1", "app.log.2" etc.
        exist, then they are renamed to "app.log.2", "app.log.3" etc.
        respectively.

        If maxBytes is zero, rollover never occurs.

        On Windows, it is not possible to rename a file that is currently opened
        by another process.  This means that it is not possible to rotate the
        log files if multiple processes is using the same log file.  In this
        case, the current log file will continue to grow until the rotation can
        be completed successfully.  In order for rotation to be possible, all of
        the other processes need to close the file first.  A mechanism, called
        "degraded" mode, has been created for this scenario.  In degraded mode,
        the log file is closed after each log message is written.  So once all
        processes have entered degraded mode, the next rotate log attempt should
        be successful and then normal logging can be resumed.

        This log handler assumes that all concurrent processes logging to a
        single file will are using only this class, and that the exact same
        parameters are provided to each instance of this class.  If, for
        example, two different processes are using this class, but with
        different values for 'maxBytes' or 'backupCount', then odd behavior is
        expected. The same is true if this class is used by one application, but
        the RotatingFileHandler is used by another.

        NOTE:  You should always provide 'filename' as an absolute path, since
        this class will need to re-open the file during rotation. If your
        application call os.chdir() then subsequent log files could be created
        in the wrong directory.
        """
        if not os.path.isabs(filename):
            if FORCE_ABSOLUTE_PATH or not os.path.split(filename)[0]:
                filename = os.path.abspath(filename)
            elif not supress_abs_warn:
                from warnings import warn
                warn("The given 'filename' should be an absolute path.  If your application calls os.chdir(), your logs may get messed up. Use 'supress_abs_warn=True' to hide this message.")
        try:
            BaseRotatingHandler.__init__(self, filename, mode, encoding)
        except TypeError:
            BaseRotatingHandler.__init__(self, filename, mode)
            self.encoding = encoding
        self._rotateFailed = False
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.lock_file = '%s.lock' % filename
        self.stream_lock = SoftFileLock(self.lock_file)
        if debug:
            self._degrade = self._degrade_debug

    def _openFile(self, mode):
        if self.encoding:
            self.stream = codecs.open(self.baseFilename, mode, self.encoding)
        else:
            self.stream = open(self.baseFilename, mode)

    def acquire(self):
        """Acquire thread and file locks. Also re-opening log file when running
        in 'degraded' mode."""
        Handler.acquire(self)
        self.stream_lock.acquire()
        if self.stream.closed:
            self._openFile(self.mode)

    def release(self):
        """Release file and thread locks. Flush stream and take care of closing
        stream in 'degraded' mode."""
        try:
            if not self.stream.closed:
                self.stream.flush()
                if self._rotateFailed:
                    self.stream.close()
        except IOError:
            if self._rotateFailed:
                self.stream.close()
        finally:
            try:
                self.stream_lock.release()
            finally:
                Handler.release(self)

    def close(self):
        """
        Closes the stream.
        """
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()
        Handler.close(self)

    def flush(self):
        """flush():  Do nothing.

        Since a flush is issued in release(), we don't do it here. To do a flush
        here, it would be necessary to re-lock everything, and it is just easier
        and cleaner to do it all in release(), rather than requiring two lock
        ops per handle() call.

        Doing a flush() here would also introduces a window of opportunity for
        another process to write to the log file in between calling
        stream.write() and stream.flush(), which seems like a bad thing."""
        pass

    def _degrade(self, degrade, msg, *args):
        """Set degrade mode or not.  Ignore msg."""
        self._rotateFailed = degrade
        del msg, args

    def _degrade_debug(self, degrade, msg, *args):
        """A more colorful version of _degade(). (This is enabled by passing
        "debug=True" at initialization).
        """
        if degrade:
            if not self._rotateFailed:
                sys.stderr.write('Degrade mode - ENTERING - (pid=%d)  %s\n' % (os.getpid(), msg % args))
                self._rotateFailed = True
        elif self._rotateFailed:
            sys.stderr.write('Degrade mode - EXITING  - (pid=%d)   %s\n' % (os.getpid(), msg % args))
            self._rotateFailed = False

    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        if self.backupCount <= 0:
            self.stream.close()
            self._openFile('w')
            return
        self.stream.close()
        try:
            tmpname = None
            while not tmpname or os.path.exists(tmpname):
                tmpname = '%s.rotate.%08d' % (self.baseFilename, randint(0, 99999999))
            try:
                os.rename(self.baseFilename, tmpname)
            except (IOError, OSError):
                exc_value = sys.exc_info()[1]
                self._degrade(True, 'rename failed.  File in use?  exception=%s', exc_value)
                return
            for i in range(self.backupCount - 1, 0, -1):
                sfn = '%s.%d' % (self.baseFilename, i)
                dfn = '%s.%d' % (self.baseFilename, i + 1)
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.baseFilename + '.1'
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(tmpname, dfn)
            self._degrade(False, 'Rotation completed')
        finally:
            self._openFile(self.mode)

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        For those that are keeping track. This differs from the standard
        library's RotatingLogHandler class. Because there is no promise to keep
        the file size under maxBytes we ignore the length of the current record.
        """
        del record
        if self._shouldRollover():
            self.stream.close()
            self._openFile(self.mode)
            return self._shouldRollover()
        return False

    def _shouldRollover(self):
        if self.maxBytes > 0:
            try:
                self.stream.seek(0, 2)
            except IOError:
                return True
            if self.stream.tell() >= self.maxBytes:
                return True
            else:
                self._degrade(False, 'Rotation done or not needed at this time')
        return False