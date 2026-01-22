import io
import json
import logging
import os
import select
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
class FileTimerServer:
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        file_path: str, the path of a FIFO special file to be created.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """

    def __init__(self, file_path: str, max_interval: float=10, daemon: bool=True, log_event: Optional[Callable[[str, Optional[FileTimerRequest]], None]]=None) -> None:
        self._file_path = file_path
        self._max_interval = max_interval
        self._daemon = daemon
        self._timers: Dict[Tuple[int, str], FileTimerRequest] = {}
        self._stop_signaled = False
        self._watchdog_thread: Optional[threading.Thread] = None
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        os.mkfifo(self._file_path)
        self._request_count = 0
        self._run_once = False
        self._log_event = log_event if log_event is not None else lambda name, request: None

    def start(self) -> None:
        log.info('Starting %s... max_interval=%s, daemon=%s', type(self).__name__, self._max_interval, self._daemon)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=self._daemon)
        log.info('Starting watchdog thread...')
        self._watchdog_thread.start()
        self._log_event('watchdog started', None)

    def stop(self) -> None:
        log.info('Stopping %s', type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            log.info('Stopping watchdog thread...')
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            log.info('No watchdog thread running, doing nothing')
        if os.path.exists(self._file_path):
            os.remove(self._file_path)
        self._log_event('watchdog stopped', None)

    def run_once(self) -> None:
        self._run_once = True
        if self._watchdog_thread:
            log.info('Stopping watchdog thread...')
            self._watchdog_thread.join()
            self._watchdog_thread = None
        else:
            log.info('No watchdog thread running, doing nothing')
        if os.path.exists(self._file_path):
            os.remove(self._file_path)

    def _watchdog_loop(self) -> None:
        with open(self._file_path) as fd:
            while not self._stop_signaled:
                try:
                    run_once = self._run_once
                    self._run_watchdog(fd)
                    if run_once:
                        break
                except Exception:
                    log.exception('Error running watchdog')

    def _run_watchdog(self, fd: io.TextIOWrapper) -> None:
        timer_requests = self._get_requests(fd, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_pids = set()
        for worker_pid, expired_timers in self.get_expired_timers(now).items():
            log.info('Reaping worker_pid=[%s]. Expired timers: %s', worker_pid, self._get_scopes(expired_timers))
            reaped_worker_pids.add(worker_pid)
            expired_timers.sort(key=lambda timer: timer.expiration_time)
            signal = 0
            expired_timer = None
            for timer in expired_timers:
                self._log_event('timer expired', timer)
                if timer.signal > 0:
                    signal = timer.signal
                    expired_timer = timer
                    break
            if signal <= 0:
                log.info('No signal specified with worker=[%s]. Do not reap it.', worker_pid)
                continue
            if self._reap_worker(worker_pid, signal):
                log.info('Successfully reaped worker=[%s] with signal=%s', worker_pid, signal)
                self._log_event('kill worker process', expired_timer)
            else:
                log.error('Error reaping worker=[%s]. Will retry on next watchdog.', worker_pid)
        self.clear_timers(reaped_worker_pids)

    def _get_scopes(self, timer_requests: List[FileTimerRequest]) -> List[str]:
        return [r.scope_id for r in timer_requests]

    def _get_requests(self, fd: io.TextIOWrapper, max_interval: float) -> List[FileTimerRequest]:
        start = time.time()
        requests = []
        while not self._stop_signaled or self._run_once:
            json_request = fd.readline()
            if len(json_request) == 0:
                if self._run_once:
                    break
                time.sleep(min(max_interval, 1))
            else:
                request = json.loads(json_request)
                pid = request['pid']
                scope_id = request['scope_id']
                expiration_time = request['expiration_time']
                signal = request['signal']
                requests.append(FileTimerRequest(worker_pid=pid, scope_id=scope_id, expiration_time=expiration_time, signal=signal))
            now = time.time()
            if now - start > max_interval:
                break
        return requests

    def register_timers(self, timer_requests: List[FileTimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_pid
            scope_id = request.scope_id
            expiration_time = request.expiration_time
            self._request_count += 1
            key = (pid, scope_id)
            if expiration_time < 0:
                if key in self._timers:
                    del self._timers[key]
            else:
                self._timers[key] = request

    def clear_timers(self, worker_pids: Set[int]) -> None:
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_pids:
                del self._timers[pid, scope_id]

    def get_expired_timers(self, deadline: float) -> Dict[int, List[FileTimerRequest]]:
        expired_timers: Dict[int, List[FileTimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_pid, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_pid: int, signal: int) -> bool:
        try:
            os.kill(worker_pid, signal)
            return True
        except ProcessLookupError:
            log.info('Process with pid=%s does not exist. Skipping', worker_pid)
            return True
        except Exception:
            log.exception('Error terminating pid=%s', worker_pid)
        return False