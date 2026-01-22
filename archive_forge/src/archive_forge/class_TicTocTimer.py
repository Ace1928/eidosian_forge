import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
class TicTocTimer(object):
    """A class to calculate and report elapsed time.

    Examples:
       >>> from pyomo.common.timing import TicTocTimer
       >>> timer = TicTocTimer()
       >>> timer.tic('starting timer') # starts the elapsed time timer (from 0)
       [    0.00] starting timer
       >>> # ... do task 1
       >>> dT = timer.toc('task 1')
       [+   0.00] task 1
       >>> print("elapsed time: %0.1f" % dT)
       elapsed time: 0.0

    If no ostream or logger is provided, then output is printed to sys.stdout

    Args:
        ostream (FILE): an optional output stream to print the timing
            information
        logger (Logger): an optional output stream using the python
           logging package. Note: the timing logged using ``logger.info()``
    """

    def __init__(self, ostream=_NotSpecified, logger=None):
        if ostream is _NotSpecified and logger is not None:
            ostream = None
        self._lastTime = self._loadTime = default_timer()
        self.ostream = ostream
        self.logger = logger
        self.level = logging.INFO
        self._start_count = 0
        self._cumul = 0

    def tic(self, msg=_NotSpecified, *args, ostream=_NotSpecified, logger=_NotSpecified, level=_NotSpecified):
        """Reset the tic/toc delta timer.

        This resets the reference time from which the next delta time is
        calculated to the current time.

        Args:
            msg (str): The message to print out.  If not specified, then
                prints out "Resetting the tic/toc delta timer"; if msg
                is None, then no message is printed.
            *args (tuple): optional positional arguments used for
                %-formatting the `msg`
            ostream (FILE): an optional output stream (overrides the ostream
                provided when the class was constructed).
            logger (Logger): an optional output stream using the python
                logging package (overrides the ostream provided when the
                class was constructed). Note: timing logged using logger.info
            level (int): an optional logging output level.

        """
        self._lastTime = self._loadTime = default_timer()
        if msg is _NotSpecified:
            msg = 'Resetting the tic/toc delta timer'
        if msg is not None:
            if args and '%' not in msg:
                deprecation_warning("tic(): 'ostream' and 'logger' should be specified as keyword arguments", version='6.4.2', logger=__package__)
                ostream, *args = args
                if args:
                    logger, *args = args
            self.toc(msg, *args, delta=False, ostream=ostream, logger=logger, level=level)

    def toc(self, msg=_NotSpecified, *args, delta=True, ostream=_NotSpecified, logger=_NotSpecified, level=_NotSpecified):
        """Print out the elapsed time.

        This resets the reference time from which the next delta time is
        calculated to the current time.

        Args:
            msg (str): The message to print out.  If not specified, then
                print out the file name, line number, and function that
                called this method; if `msg` is None, then no message is
                printed.
            *args (tuple): optional positional arguments used for
                %-formatting the `msg`
            delta (bool): print out the elapsed wall clock time since
                the last call to :meth:`tic` (``False``) or since the
                most recent call to either :meth:`tic` or :meth:`toc`
                (``True`` (default)).
            ostream (FILE): an optional output stream (overrides the ostream
                provided when the class was constructed).
            logger (Logger): an optional output stream using the python
                logging package (overrides the ostream provided when the
                class was constructed). Note: timing logged using `level`
            level (int): an optional logging output level.
        """
        if msg is _NotSpecified:
            msg = 'File "%s", line %s in %s' % traceback.extract_stack(limit=2)[0][:3]
        if args and msg is not None and ('%' not in msg):
            deprecation_warning("toc(): 'delta', 'ostream', and 'logger' should be specified as keyword arguments", version='6.4.2', logger=__package__)
            delta, *args = args
            if args:
                ostream, *args = args
            if args:
                logger, *args = args
        now = default_timer()
        if self._start_count or self._lastTime is None:
            ans = self._cumul
            if self._lastTime:
                ans += default_timer() - self._lastTime
            if msg is not None:
                fmt = '[%8.2f|%4d] %s'
                data = (ans, self._start_count, msg)
        elif delta:
            ans = now - self._lastTime
            self._lastTime = now
            if msg is not None:
                fmt = '[+%7.2f] %s'
                data = (ans, msg)
        else:
            ans = now - self._loadTime
            self._lastTime = now
            if msg is not None:
                fmt = '[%8.2f] %s'
                data = (ans, msg)
        if msg is not None:
            if logger is _NotSpecified:
                logger = self.logger
            if logger is not None:
                if level is _NotSpecified:
                    level = self.level
                logger.log(level, GeneralTimer(fmt, data), *args)
            if ostream is _NotSpecified:
                ostream = self.ostream
                if ostream is _NotSpecified:
                    if logger is None:
                        ostream = sys.stdout
                    else:
                        ostream = None
            if ostream is not None:
                msg = fmt % data
                if args:
                    msg = msg % args
                ostream.write(msg + '\n')
        return ans

    def stop(self):
        delta, self._lastTime = (self._lastTime, None)
        if delta is None:
            raise RuntimeError('Stopping a TicTocTimer that was already stopped')
        delta = default_timer() - delta
        self._cumul += delta
        return delta

    def start(self):
        if self._lastTime:
            self.stop()
        self._start_count += 1
        self._lastTime = default_timer()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, et, ev, tb):
        self.stop()