import datetime
import errno
import logging
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from io import TextIOWrapper
from logging.handlers import BaseRotatingHandler, TimedRotatingFileHandler
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple
from portalocker import LOCK_EX, lock, unlock
import logging.handlers  # noqa: E402
class ConcurrentTimedRotatingFileHandler(TimedRotatingFileHandler):
    """A time-based rotating log handler that supports concurrent access across
    multiple processes or hosts (using logs on a shared network drive).

    You can also include size-based rotation by setting maxBytes > 0.
    WARNING: if you only want time-based rollover and NOT also size-based, set maxBytes=0,
    which is already the default.
    Please note that when size-based rotation is done, it still uses the naming scheme
    of the time-based rotation. If multiple rotations had to be done within the timeframe of
    the time-based rollover name, then a number like ".1" will be appended to the end of the name.

    Note that `errors` is ignored unless using Python 3.9 or later.
    """

    def __init__(self, filename: str, when: str='h', interval: int=1, backupCount: int=0, encoding: Optional[str]=None, delay: bool=False, utc: bool=False, atTime: Optional[datetime.time]=None, errors: Optional[str]=None, maxBytes: int=0, use_gzip: bool=False, owner: Optional[Tuple[str, str]]=None, chmod: Optional[int]=None, umask: Optional[int]=None, newline: Optional[str]=None, terminator: str='\n', unicode_error_policy: str='ignore', lock_file_directory: Optional[str]=None, **kwargs):
        if 'mode' in kwargs:
            del kwargs['mode']
        trfh_kwargs: Dict[str, Optional[str]] = {}
        if sys.version_info >= (3, 9):
            trfh_kwargs['errors'] = errors
        TimedRotatingFileHandler.__init__(self, filename, when=when, interval=interval, backupCount=backupCount, encoding=encoding, delay=delay, utc=utc, atTime=atTime, **trfh_kwargs)
        self.clh = ConcurrentRotatingFileHandler(filename, mode='a', backupCount=backupCount, encoding=encoding, delay=None, maxBytes=maxBytes, use_gzip=use_gzip, owner=owner, chmod=chmod, umask=umask, newline=newline, terminator=terminator, unicode_error_policy=unicode_error_policy, lock_file_directory=lock_file_directory, **kwargs)
        self.num_rollovers = 0
        self.__internal_close()
        self.initialize_rollover_time()

    def __internal_close(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None

    def _console_log(self, msg: str, stack: bool=False) -> None:
        self.clh._console_log(msg, stack=stack)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.

        Override from parent class to handle file locking for the duration of rollover and write.
        This also does the formatting *before* locks are obtained, in case the format itself does
        logging calls from within. Rollover also occurs while the lock is held.
        """
        try:
            msg = self.format(record)
            try:
                self.clh._do_lock()
                try:
                    if self.shouldRollover(record):
                        self.doRollover()
                except Exception as e:
                    self._console_log('Unable to do rollover: {}\n{}'.format(e, traceback.format_exc()))
                self.clh.do_write(msg)
            finally:
                self.clh._do_unlock()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

    def read_rollover_time(self) -> None:
        lock_file = self.clh.stream_lock
        if not lock_file or not self.clh.is_locked:
            self._console_log('No rollover time (lock) file to read from. Lock is not held?')
            return
        try:
            lock_file.seek(0)
            raw_time = lock_file.read()
        except OSError:
            self.rolloverAt = 0
            self._console_log(f"Couldn't read rollover time from file {lock_file!r}")
            return
        try:
            self.rolloverAt = int(raw_time.strip())
        except ValueError:
            self.rolloverAt = 0
            self._console_log(f"Couldn't read rollover time from file: {raw_time!r}")

    def write_rollover_time(self) -> None:
        """Write the next rollover time (current value of self.rolloverAt) to the lock file."""
        lock_file = self.clh.stream_lock
        if not lock_file or not self.clh.is_locked:
            self._console_log('No rollover time (lock) file to write to. Lock is not held?')
            return
        lock_file.seek(0)
        lock_file.write(str(self.rolloverAt))
        lock_file.truncate()
        lock_file.flush()
        os.fsync(lock_file.fileno())
        self._console_log(f'Wrote rollover time: {self.rolloverAt}')

    def initialize_rollover_time(self) -> None:
        """Run by the __init__ to read an existing rollover time from the lockfile,
        and if it can't do that, compute and write a new one."""
        try:
            self.clh._do_lock()
            self.read_rollover_time()
            self._console_log(f'Initializing; reading rollover time: {self.rolloverAt}')
            if self.rolloverAt != 0:
                return
            current_time = int(time.time())
            new_rollover_at = self.computeRollover(current_time)
            while new_rollover_at <= current_time:
                new_rollover_at += self.interval
            self.rolloverAt = new_rollover_at
            self.write_rollover_time()
            self._console_log(f'Set initial rollover time: {self.rolloverAt}')
        finally:
            self.clh._do_unlock()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """Determine if the rollover should occur."""
        self.read_rollover_time()
        do_rollover = False
        if super(ConcurrentTimedRotatingFileHandler, self).shouldRollover(record):
            self._console_log('Rolling over because of time')
            do_rollover = True
        elif self.clh.shouldRollover(record):
            self.clh._console_log('Rolling over because of size')
            do_rollover = True
        if do_rollover:
            return True
        return False

    def doRollover(self) -> None:
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.

        This code was adapted from the TimedRotatingFileHandler class from Python 3.11.
        """
        self.clh._close()
        self.__internal_close()
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                addend = 3600 if dstNow else -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + '.' + time.strftime(self.suffix, timeTuple))
        gzip_ext = '.gz' if self.clh.use_gzip else ''
        counter = 1
        if os.path.exists(dfn + gzip_ext):
            while os.path.exists(f'{dfn}.{counter}{gzip_ext}'):
                ending = f'.{counter - 1}{gzip_ext}'
                if dfn.endswith(ending):
                    dfn = dfn[:-len(ending)]
                counter += 1
            dfn = f'{dfn}.{counter}'
        self.rotate(self.baseFilename, dfn)
        if self.clh.use_gzip:
            self.clh.do_gzip(dfn)
        if self.backupCount > 0:
            for file in self.getFilesToDelete():
                os.remove(file)
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and (not self.utc):
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:
                    addend = -3600
                else:
                    addend = 3600
                newRolloverAt += addend
        self.num_rollovers += 1
        self.rolloverAt = newRolloverAt
        self.write_rollover_time()
        self._console_log(f'Rotation completed (on time) {dfn}')

    def getFilesToDelete(self) -> List[str]:
        """
        Determine the files to delete when rolling over.

        Copied from Python 3.11, and only applied when the current Python
        seems to be Python 3.8 or lower, which is when this seemed to change.
        The newer version supports custom suffixes like ours, such
        as when hitting a size limit before the time limit.
        """
        if sys.version_info >= (3, 9):
            return super().getFilesToDelete()
        dirName, baseName = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        n, e = os.path.splitext(baseName)
        prefix = n + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if self.namer is None:
                if not fileName.startswith(baseName):
                    continue
            elif not fileName.startswith(baseName) and fileName.endswith(e) and (len(fileName) > plen + 1) and (not fileName[plen + 1].isdigit()):
                continue
            if fileName[:plen] == prefix:
                suffix = fileName[plen:]
                parts = suffix.split('.')
                for part in parts:
                    if self.extMatch.match(part):
                        result.append(os.path.join(dirName, fileName))
                        break
        if len(result) < self.backupCount:
            result = []
        else:
            result.sort()
            result = result[:len(result) - self.backupCount]
        return result