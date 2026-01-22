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
class TBDirWatcher:

    def __init__(self, tbwatcher: 'TBWatcher', logdir: str, save: bool, namespace: Optional[str], queue: 'PriorityQueue', force: bool=False) -> None:
        self.directory_watcher = util.get_module('tensorboard.backend.event_processing.directory_watcher', required='Please install tensorboard package')
        self.tf_compat = util.get_module('tensorboard.compat', required='Please install tensorboard package')
        self._tbwatcher = tbwatcher
        self._generator = self.directory_watcher.DirectoryWatcher(logdir, self._loader(save, namespace), self._is_our_tfevents_file)
        self._thread = threading.Thread(target=self._thread_except_body)
        self._first_event_timestamp = None
        self._shutdown = threading.Event()
        self._queue = queue
        self._file_version = None
        self._namespace = namespace
        self._logdir = logdir
        self._hostname = socket.gethostname()
        self._force = force
        self._process_events_lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def _is_our_tfevents_file(self, path: str) -> bool:
        """Check if a path has been modified since launch and contains tfevents."""
        if not path:
            raise ValueError('Path must be a nonempty string')
        path = self.tf_compat.tf.compat.as_str_any(path)
        if self._force:
            return is_tfevents_file_created_by(path, None, None)
        else:
            return is_tfevents_file_created_by(path, self._hostname, self._tbwatcher._settings._start_time)

    def _loader(self, save: bool=True, namespace: Optional[str]=None) -> 'EventFileLoader':
        """Incredibly hacky class generator to optionally save / prefix tfevent files."""
        _loader_interface = self._tbwatcher._interface
        _loader_settings = self._tbwatcher._settings
        try:
            from tensorboard.backend.event_processing import event_file_loader
        except ImportError:
            raise Exception('Please install tensorboard package')

        class EventFileLoader(event_file_loader.EventFileLoader):

            def __init__(self, file_path: str) -> None:
                super().__init__(file_path)
                if save:
                    if REMOTE_FILE_TOKEN in file_path:
                        logger.warning('Not persisting remote tfevent file: %s', file_path)
                    else:
                        logdir = os.path.dirname(file_path)
                        parts = list(os.path.split(logdir))
                        if namespace and parts[-1] == namespace:
                            parts.pop()
                            logdir = os.path.join(*parts)
                        _link_and_save_file(path=file_path, base_path=logdir, interface=_loader_interface, settings=_loader_settings)
        return EventFileLoader

    def _process_events(self, shutdown_call: bool=False) -> None:
        try:
            with self._process_events_lock:
                for event in self._generator.Load():
                    self.process_event(event)
        except (self.directory_watcher.DirectoryDeletedError, StopIteration, RuntimeError, OSError) as e:
            logger.debug('Encountered tensorboard directory watcher error: %s', e)
            if not self._shutdown.is_set() and (not shutdown_call):
                time.sleep(ERROR_DELAY)

    def _thread_except_body(self) -> None:
        try:
            self._thread_body()
        except Exception as e:
            logger.exception('generic exception in TBDirWatcher thread')
            raise e

    def _thread_body(self) -> None:
        """Check for new events every second."""
        shutdown_time: Optional[float] = None
        while True:
            self._process_events()
            if self._shutdown.is_set():
                now = time.time()
                if not shutdown_time:
                    shutdown_time = now + SHUTDOWN_DELAY
                elif now > shutdown_time:
                    break
            time.sleep(1)

    def process_event(self, event: 'ProtoEvent') -> None:
        if self._first_event_timestamp is None:
            self._first_event_timestamp = event.wall_time
        if event.HasField('file_version'):
            self._file_version = event.file_version
        if event.HasField('summary'):
            self._queue.put(Event(event, self._namespace))

    def shutdown(self) -> None:
        self._process_events(shutdown_call=True)
        self._shutdown.set()

    def finish(self) -> None:
        self.shutdown()
        self._thread.join()