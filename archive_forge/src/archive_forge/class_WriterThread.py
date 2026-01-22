import atexit
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
import psutil
import wandb
from ..interface.interface_queue import InterfaceQueue
from ..lib import tracelog
from . import context, handler, internal_util, sender, writer
class WriterThread(internal_util.RecordLoopThread):
    """Read records from queue and dispatch to writer routines."""
    _record_q: 'Queue[Record]'
    _result_q: 'Queue[Result]'
    _context_keeper: context.ContextKeeper

    def __init__(self, settings: 'SettingsStatic', record_q: 'Queue[Record]', result_q: 'Queue[Result]', stopped: 'Event', interface: 'InterfaceQueue', sender_q: 'Queue[Record]', context_keeper: context.ContextKeeper, debounce_interval_ms: 'float'=1000) -> None:
        super().__init__(input_record_q=record_q, result_q=result_q, stopped=stopped, debounce_interval_ms=debounce_interval_ms)
        self.name = 'WriterThread'
        self._settings = settings
        self._record_q = record_q
        self._result_q = result_q
        self._sender_q = sender_q
        self._interface = interface
        self._context_keeper = context_keeper

    def _setup(self) -> None:
        self._wm = writer.WriteManager(settings=self._settings, record_q=self._record_q, result_q=self._result_q, sender_q=self._sender_q, interface=self._interface, context_keeper=self._context_keeper)

    def _process(self, record: 'Record') -> None:
        self._wm.write(record)

    def _finish(self) -> None:
        self._wm.finish()

    def _debounce(self) -> None:
        self._wm.debounce()