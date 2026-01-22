import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.multi_threaded_pg import (
class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    TEST_ERROR_EXIT_CODE = 10

    def _should_stop_test_suite(self) -> bool:
        return False

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE

    def join_or_run(self, fn):

        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()
        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str='runTest') -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        self.pid_to_pipe = {}

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()
        self.processes = []

    def _current_test_name(self) -> str:
        return self.id().split('.')[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []
        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(target=self.__class__._run, name='process ' + str(rank), args=(rank, self._current_test_name(), self.file_name, child_conn))
            process.start()
            logger.info('Started process %s with pid %s', rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context('spawn').Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info('Starting event listener thread for rank %s', rank)
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])
            if parent_pipe in ready_pipes:
                if parent_pipe.closed:
                    logger.info('Pipe closed for process %s, stopping event listener thread', rank)
                    return
                event = parent_pipe.recv()
                logger.info('Received event %s on process %s', event, rank)
                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    with tempfile.NamedTemporaryFile(mode='r+') as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())
                        logger.info('Process %s sent traceback', rank)
            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(target=MultiProcessTestCase._event_listener, args=(parent_pipe, signal_recv_pipe, self.rank), daemon=True)
        event_listener_thread.start()
        if sys.platform != 'win32' and sys.platform != 'darwin':
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info('Process %s skipping test %s for following reason: %s', self.rank, test_name, str(se))
            sys.exit(TEST_SKIPS['generic'].exit_code)
        except Exception as e:
            logger.error('Caught exception: \n%s exiting process %s with exit code: %s', traceback.format_exc(), self.rank, MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)
            assert event_listener_thread is not None
            event_listener_thread.join()
            parent_pipe.close()

    def _get_timedout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error('Encountered error while trying to get traceback for process %s: %s', i, e)
        for rank, pipe in pipes:
            try:
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info('Pipe closed for process %s, cannot retrieve traceback', rank)
                        continue
                    traceback = pipe.recv()
                    logger.error('Process %s timed out with traceback: \n\n%s', rank, traceback)
                else:
                    logger.error('Could not retrieve traceback for timed out process: %s', rank)
            except ConnectionError as e:
                logger.error('Encountered error while trying to get traceback for process %s: %s', rank, e)

    def _join_processes(self, fn) -> None:
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                for i, p in enumerate(self.processes):
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(f'Process {i} terminated with exit code {p.exitcode}, terminating remaining processes.')
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                if all((p.exitcode is not None for p in self.processes)):
                    break
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(f'Timing out after {timeout} seconds and killing subprocesses.')
                    for p in self.processes:
                        p.terminate()
                    break
                time.sleep(0.1)
            elapsed_time = time.time() - start_time
            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f'Process {i} timed out after {elapsed_time} seconds')
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        if not self.processes:
            logger.warning('Note: no subprocesses were spawned, test was likely skipped.')
            return
        first_process = self.processes[0]
        errored_processes = [(i, p) for i, p in enumerate(self.processes) if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE]
        if errored_processes:
            error = ''
            for i, process in errored_processes:
                error_message = self.pid_to_pipe[process.pid].recv()
                error += 'Process {} exited with error code {} and exception:\n{}\n'.format(i, MultiProcessTestCase.TEST_ERROR_EXIT_CODE, error_message)
            raise RuntimeError(error)
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(f'Process {i} terminated or timed out after {elapsed_time} seconds')
            self.assertEqual(p.exitcode, first_process.exitcode, msg='Expect process {} exit code to match Process 0 exit code of {}, but got {}'.format(i, first_process.exitcode, p.exitcode))
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:
                    logger.info('Skipping %s on sandcastle for the following reason: %s', self.id(), skip.message)
                    return
                else:
                    raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0, msg=f'Expected zero exit code but got {first_process.exitcode} for pid: {first_process.pid}')

    @property
    def is_master(self) -> bool:
        return self.rank == 0