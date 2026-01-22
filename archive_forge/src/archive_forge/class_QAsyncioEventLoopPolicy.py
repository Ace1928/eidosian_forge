from PySide6.QtCore import (QCoreApplication, QDateTime, QDeadlineTimer,
from . import futures
from . import tasks
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import enum
import os
import signal
import socket
import subprocess
import typing
import warnings
class QAsyncioEventLoopPolicy(asyncio.AbstractEventLoopPolicy):

    def __init__(self, application: typing.Optional[QCoreApplication]=None, quit_qapp: bool=True) -> None:
        super().__init__()
        if application is None:
            if QCoreApplication.instance() is None:
                application = QCoreApplication()
            else:
                application = QCoreApplication.instance()
        self._application: QCoreApplication = application
        self._quit_qapp = quit_qapp
        self._event_loop: typing.Optional[asyncio.AbstractEventLoop] = None
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if self._event_loop is None:
            self._event_loop = QAsyncioEventLoop(self._application)
        return self._event_loop

    def set_event_loop(self, loop: typing.Optional[asyncio.AbstractEventLoop]) -> None:
        self._event_loop = loop

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return QAsyncioEventLoop(self._application, quit_qapp=self._quit_qapp)

    def get_child_watcher(self) -> asyncio.AbstractChildWatcher:
        raise DeprecationWarning('Child watchers are deprecated since Python 3.12')

    def set_child_watcher(self, watcher: asyncio.AbstractChildWatcher) -> None:
        raise DeprecationWarning('Child watchers are deprecated since Python 3.12')