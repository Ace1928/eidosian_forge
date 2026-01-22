import glob
import logging
import os
import queue
import socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib import filesystem
from wandb.viz import CustomChart
from . import run as internal_run
class TBEventConsumer:
    """Consume tfevents from a priority queue.

    There should always only be one of these per run_manager.  We wait for 10 seconds of
    queued events to reduce the chance of multiple tfevent files triggering out of order
    steps.
    """

    def __init__(self, tbwatcher: TBWatcher, queue: 'PriorityQueue', run_proto: 'RunRecord', settings: 'SettingsStatic', delay: int=10) -> None:
        self._tbwatcher = tbwatcher
        self._queue = queue
        self._thread = threading.Thread(target=self._thread_except_body)
        self._shutdown = threading.Event()
        self.tb_history = TBHistory()
        self._delay = delay

        def datatypes_cb(fname: GlobStr) -> None:
            files: FilesDict = dict(files=[(fname, 'now')])
            self._tbwatcher._interface.publish_files(files)
        self._internal_run = internal_run.InternalRun(run_proto, settings, datatypes_cb)
        self._internal_run._set_internal_run_interface(self._tbwatcher._interface)

    def start(self) -> None:
        self._start_time = time.time()
        self._thread.start()

    def finish(self) -> None:
        self._delay = 0
        self._shutdown.set()
        self._thread.join()
        while not self._queue.empty():
            event = self._queue.get(True, 1)
            if event:
                self._handle_event(event, history=self.tb_history)
                items = self.tb_history._get_and_reset()
                for item in items:
                    self._save_row(item)

    def _thread_except_body(self) -> None:
        try:
            self._thread_body()
        except Exception as e:
            logger.exception('generic exception in TBEventConsumer thread')
            raise e

    def _thread_body(self) -> None:
        while True:
            try:
                event = self._queue.get(True, 1)
                if time.time() < self._start_time + self._delay and (not self._shutdown.is_set()):
                    self._queue.put(event)
                    time.sleep(0.1)
                    continue
            except queue.Empty:
                event = None
                if self._shutdown.is_set():
                    break
            if event:
                self._handle_event(event, history=self.tb_history)
                items = self.tb_history._get_and_reset()
                for item in items:
                    self._save_row(item)
        self.tb_history._flush()
        items = self.tb_history._get_and_reset()
        for item in items:
            self._save_row(item)

    def _handle_event(self, event: 'ProtoEvent', history: Optional['TBHistory']=None) -> None:
        wandb.tensorboard._log(event.event, step=event.event.step, namespace=event.namespace, history=history)

    def _save_row(self, row: 'HistoryDict') -> None:
        chart_keys = set()
        for k in row:
            if isinstance(row[k], CustomChart):
                chart_keys.add(k)
                key = row[k].get_config_key(k)
                value = row[k].get_config_value('Vega2', row[k].user_query(f'{k}_table'))
                row[k] = row[k]._data
                self._tbwatcher._interface.publish_config(val=value, key=key)
        for k in chart_keys:
            row[f'{k}_table'] = row.pop(k)
        self._tbwatcher._interface.publish_history(row, run=self._internal_run, publish_step=False)