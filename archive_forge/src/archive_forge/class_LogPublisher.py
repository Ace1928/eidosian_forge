import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
class LogPublisher:
    """
    Class for singleton log message publishing.
    """
    synchronized = ['msg']

    def __init__(self, observerPublisher=None, publishPublisher=None, logBeginner=None, warningsModule=warnings):
        if publishPublisher is None:
            publishPublisher = NewPublisher()
            if observerPublisher is None:
                observerPublisher = publishPublisher
        if observerPublisher is None:
            observerPublisher = NewPublisher()
        self._observerPublisher = observerPublisher
        self._publishPublisher = publishPublisher
        self._legacyObservers = []
        if logBeginner is None:
            beginnerPublisher = NewPublisher()
            beginnerPublisher.addObserver(observerPublisher)
            logBeginner = LogBeginner(beginnerPublisher, cast(BinaryIO, NullFile()), sys, warnings)
        self._logBeginner = logBeginner
        self._warningsModule = warningsModule
        self._oldshowwarning = warningsModule.showwarning
        self.showwarning = self._logBeginner.showwarning

    @property
    def observers(self):
        """
        Property returning all observers registered on this L{LogPublisher}.

        @return: observers previously added with L{LogPublisher.addObserver}
        @rtype: L{list} of L{callable}
        """
        return [x.legacyObserver for x in self._legacyObservers]

    def _startLogging(self, other, setStdout):
        """
        Begin logging to the L{LogBeginner} associated with this
        L{LogPublisher}.

        @param other: the observer to log to.
        @type other: L{LogBeginner}

        @param setStdout: if true, send standard I/O to the observer as well.
        @type setStdout: L{bool}
        """
        wrapped = LegacyLogObserverWrapper(other)
        self._legacyObservers.append(wrapped)
        self._logBeginner.beginLoggingTo([wrapped], True, setStdout)

    def _stopLogging(self):
        """
        Clean-up hook for fixing potentially global state.  Only for testing of
        this module itself.  If you want less global state, use the new
        warnings system in L{twisted.logger}.
        """
        if self._warningsModule.showwarning == self.showwarning:
            self._warningsModule.showwarning = self._oldshowwarning

    def addObserver(self, other):
        """
        Add a new observer.

        @type other: Provider of L{ILogObserver}
        @param other: A callable object that will be called with each new log
            message (a dict).
        """
        wrapped = LegacyLogObserverWrapper(other)
        self._legacyObservers.append(wrapped)
        self._observerPublisher.addObserver(wrapped)

    def removeObserver(self, other):
        """
        Remove an observer.
        """
        for observer in self._legacyObservers:
            if observer.legacyObserver == other:
                self._legacyObservers.remove(observer)
                self._observerPublisher.removeObserver(observer)
                break

    def msg(self, *message, **kw):
        """
        Log a new message.

        The message should be a native string, i.e. bytes on Python 2 and
        Unicode on Python 3. For compatibility with both use the native string
        syntax, for example::

            >>> log.msg('Hello, world.')

        You MUST avoid passing in Unicode on Python 2, and the form::

            >>> log.msg('Hello ', 'world.')

        This form only works (sometimes) by accident.

        Keyword arguments will be converted into items in the event
        dict that is passed to L{ILogObserver} implementations.
        Each implementation, in turn, can define keys that are used
        by it specifically, in addition to common keys listed at
        L{ILogObserver.__call__}.

        For example, to set the C{system} parameter while logging
        a message::

        >>> log.msg('Started', system='Foo')

        """
        actualEventDict = cast(EventDict, (context.get(ILogContext) or {}).copy())
        actualEventDict.update(kw)
        actualEventDict['message'] = message
        actualEventDict['time'] = time.time()
        if 'isError' not in actualEventDict:
            actualEventDict['isError'] = 0
        _publishNew(self._publishPublisher, actualEventDict, textFromEventDict)