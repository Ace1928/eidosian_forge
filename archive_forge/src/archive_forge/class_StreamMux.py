import functools
import multiprocessing
import queue
import threading
import time
from threading import Event
from typing import Any, Callable, Dict, List, Optional
import psutil
import wandb
import wandb.util
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import (
from wandb.sdk.lib.printer import get_printer
from wandb.sdk.wandb_run import Run
from ..interface.interface_relay import InterfaceRelay
class StreamMux:
    _streams_lock: threading.Lock
    _streams: Dict[str, StreamRecord]
    _port: Optional[int]
    _pid: Optional[int]
    _action_q: 'queue.Queue[StreamAction]'
    _stopped: Event
    _pid_checked_ts: Optional[float]
    _mailbox: Mailbox

    def __init__(self) -> None:
        self._streams_lock = threading.Lock()
        self._streams = dict()
        self._port = None
        self._pid = None
        self._stopped = Event()
        self._action_q = queue.Queue()
        self._pid_checked_ts = None
        self._mailbox = Mailbox()
        self._mailbox.enable_keepalive()

    def _get_stopped_event(self) -> 'Event':
        return self._stopped

    def set_port(self, port: int) -> None:
        self._port = port

    def set_pid(self, pid: int) -> None:
        self._pid = pid

    def add_stream(self, stream_id: str, settings: SettingsStatic) -> None:
        action = StreamAction(action='add', stream_id=stream_id, data=settings)
        self._action_q.put(action)
        action.wait_handled()

    def start_stream(self, stream_id: str) -> None:
        action = StreamAction(action='start', stream_id=stream_id)
        self._action_q.put(action)
        action.wait_handled()

    def update_stream(self, stream_id: str, settings: SettingsStatic) -> None:
        action = StreamAction(action='update', stream_id=stream_id, data=settings)
        self._action_q.put(action)
        action.wait_handled()

    def del_stream(self, stream_id: str) -> None:
        action = StreamAction(action='del', stream_id=stream_id)
        self._action_q.put(action)
        action.wait_handled()

    def drop_stream(self, stream_id: str) -> None:
        action = StreamAction(action='drop', stream_id=stream_id)
        self._action_q.put(action)
        action.wait_handled()

    def teardown(self, exit_code: int) -> None:
        action = StreamAction(action='teardown', stream_id='na', data=exit_code)
        self._action_q.put(action)
        action.wait_handled()

    def stream_names(self) -> List[str]:
        with self._streams_lock:
            names = list(self._streams.keys())
            return names

    def has_stream(self, stream_id: str) -> bool:
        with self._streams_lock:
            return stream_id in self._streams

    def get_stream(self, stream_id: str) -> StreamRecord:
        with self._streams_lock:
            stream = self._streams[stream_id]
            return stream

    def _process_add(self, action: StreamAction) -> None:
        stream = StreamRecord(action._data, mailbox=self._mailbox)
        settings = action._data
        thread = StreamThread(target=wandb.wandb_sdk.internal.internal.wandb_internal, kwargs=dict(settings=settings, record_q=stream._record_q, result_q=stream._result_q, port=self._port, user_pid=self._pid))
        stream.start_thread(thread)
        with self._streams_lock:
            self._streams[action._stream_id] = stream

    def _process_start(self, action: StreamAction) -> None:
        with self._streams_lock:
            self._streams[action._stream_id].mark_started()

    def _process_update(self, action: StreamAction) -> None:
        with self._streams_lock:
            self._streams[action._stream_id].update(action._data)

    def _process_del(self, action: StreamAction) -> None:
        with self._streams_lock:
            stream = self._streams.pop(action._stream_id)
            stream.join()

    def _process_drop(self, action: StreamAction) -> None:
        with self._streams_lock:
            if action._stream_id in self._streams:
                stream = self._streams.pop(action._stream_id)
                stream.drop()
                stream.join()

    def _on_probe_exit(self, probe_handle: MailboxProbe, stream: StreamRecord) -> None:
        handle = probe_handle.get_mailbox_handle()
        if handle:
            result = handle.wait(timeout=0, release=False)
            if not result:
                return
            probe_handle.set_probe_result(result)
        handle = stream.interface.deliver_poll_exit()
        probe_handle.set_mailbox_handle(handle)

    def _on_progress_exit(self, progress_handle: MailboxProgress) -> None:
        pass

    def _on_progress_exit_all(self, progress_all_handle: MailboxProgressAll) -> None:
        probe_handles = []
        progress_handles = progress_all_handle.get_progress_handles()
        for progress_handle in progress_handles:
            probe_handles.extend(progress_handle.get_probe_handles())
        assert probe_handles
        if self._check_orphaned():
            self._stopped.set()
        poll_exit_responses: List[Optional[pb.PollExitResponse]] = []
        for probe_handle in probe_handles:
            result = probe_handle.get_probe_result()
            if result:
                poll_exit_responses.append(result.response.poll_exit_response)
        Run._footer_file_pusher_status_info(poll_exit_responses, printer=self._printer)

    def _finish_all(self, streams: Dict[str, StreamRecord], exit_code: int) -> None:
        if not streams:
            return
        printer = get_printer(all((stream._settings._jupyter for stream in streams.values())))
        self._printer = printer
        exit_handles = []
        started_streams: Dict[str, StreamRecord] = {}
        not_started_streams: Dict[str, StreamRecord] = {}
        for stream_id, stream in streams.items():
            d = started_streams if stream._started else not_started_streams
            d[stream_id] = stream
        for stream in started_streams.values():
            handle = stream.interface.deliver_exit(exit_code)
            handle.add_progress(self._on_progress_exit)
            handle.add_probe(functools.partial(self._on_probe_exit, stream=stream))
            exit_handles.append(handle)
        got_result = self._mailbox.wait_all(handles=exit_handles, timeout=-1, on_progress_all=self._on_progress_exit_all)
        assert got_result
        for _sid, stream in started_streams.items():
            poll_exit_handle = stream.interface.deliver_poll_exit()
            server_info_handle = stream.interface.deliver_request_server_info()
            final_summary_handle = stream.interface.deliver_get_summary()
            sampled_history_handle = stream.interface.deliver_request_sampled_history()
            internal_messages_handle = stream.interface.deliver_internal_messages()
            result = internal_messages_handle.wait(timeout=-1)
            assert result
            internal_messages_response = result.response.internal_messages_response
            job_info_handle = stream.interface.deliver_request_job_info()
            result = poll_exit_handle.wait(timeout=-1)
            assert result
            poll_exit_response = result.response.poll_exit_response
            result = server_info_handle.wait(timeout=-1)
            assert result
            server_info_response = result.response.server_info_response
            result = sampled_history_handle.wait(timeout=-1)
            assert result
            sampled_history = result.response.sampled_history_response
            result = final_summary_handle.wait(timeout=-1)
            assert result
            final_summary = result.response.get_summary_response
            result = job_info_handle.wait(timeout=-1)
            assert result
            job_info = result.response.job_info_response
            Run._footer(sampled_history=sampled_history, final_summary=final_summary, poll_exit_response=poll_exit_response, server_info_response=server_info_response, internal_messages_response=internal_messages_response, job_info=job_info, settings=stream._settings, printer=printer)
            stream.join()
        for stream in not_started_streams.values():
            stream.join()

    def _process_teardown(self, action: StreamAction) -> None:
        exit_code: int = action._data
        with self._streams_lock:
            streams_copy = self._streams.copy()
        self._finish_all(streams_copy, exit_code)
        with self._streams_lock:
            self._streams = dict()
        self._stopped.set()

    def _process_action(self, action: StreamAction) -> None:
        if action._action == 'add':
            self._process_add(action)
            return
        if action._action == 'update':
            self._process_update(action)
            return
        if action._action == 'start':
            self._process_start(action)
            return
        if action._action == 'del':
            self._process_del(action)
            return
        if action._action == 'drop':
            self._process_drop(action)
            return
        if action._action == 'teardown':
            self._process_teardown(action)
            return
        raise AssertionError(f'Unsupported action: {action._action}')

    def _check_orphaned(self) -> bool:
        if not self._pid:
            return False
        time_now = time.time()
        if self._pid_checked_ts and time_now < self._pid_checked_ts + 2:
            return False
        self._pid_checked_ts = time_now
        return not psutil.pid_exists(self._pid)

    def _loop(self) -> None:
        while not self._stopped.is_set():
            if self._check_orphaned():
                self._stopped.set()
            try:
                action = self._action_q.get(timeout=1)
            except queue.Empty:
                continue
            self._process_action(action)
            action.set_handled()
            self._action_q.task_done()
        self._action_q.join()

    def loop(self) -> None:
        try:
            self._loop()
        except Exception as e:
            raise e

    def cleanup(self) -> None:
        pass