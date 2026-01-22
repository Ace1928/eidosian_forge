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
class MultiThreadedTestCase(TestCase):
    """
    Test runner that runs all tests with the in-proc process group using
    multiple threads with the threaded process group.

    Each test spawns world_size threads and run the test method in each thread.

    Difference from regular MultiProcess test runner:
    Must explicitly defines SetUp and call self._spawn_threads() to run the tests.
    Cannot use setUp / tearDown (must use perThreadSetup / perThreadShutdown)
        to set up / tear down each thread when running each test.
    No global state possible
        How bad of a limitation is this?
    """
    exception_queue = queue.Queue()
    MAIN_THREAD_RANK = -1

    def join_or_run(self, fn):

        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_THREAD_RANK:
                self._join_threads(self.threads, fn)
            else:
                fn()
        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str='runTest') -> None:
        super().__init__(method_name)
        test_fn = getattr(self, method_name, None)
        setattr(self, method_name, self.join_or_run(test_fn))

    def perThreadSetUp(self):
        pass

    def perThreadTearDown(self):
        pass

    def setUp(self) -> None:
        """
        setUp only set up things in the main thread, if you want to configure things
        in the spawned threads, use perThreadSetUp
        """
        super().setUp()
        self.rank = self.MAIN_THREAD_RANK
        self.threads = []
        os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'

    def tearDown(self):
        """
        tearDown only set up things in the main thread, if you want to configure things
        in the spawned threads, use perThreadTearDown
        """
        super().tearDown()
        self.threads = []

    def _spawn_threads(self):
        """
        class method to spawn threads and run test, use this method in the SetUp of your TestCase
        """
        test_name = self._current_test_name
        world = _install_threaded_pg()
        self.__class__.global_store = c10d.HashStore()

        def world_is_valid():
            return world == c10d.distributed_c10d._world
        if not world_is_valid():
            raise RuntimeError('Invalid world')
        for rank in range(self.world_size):
            t = threading.Thread(target=self.__class__._run, args=(test_name, rank, self.world_size))
            t.start()
            self.threads.append(t)

    @classmethod
    def _run(cls, test_name, rank, world_size):
        self = cls(test_name)
        self.rank = rank
        if hasattr(self, '_tls'):
            self._tls = threading.local()
            self._tls.precision = TestCase._precision
            self._tls.rel_tol = TestCase._rel_tol
        self.run_test_with_threaded_pg(test_name, rank, world_size)

    def run_test_with_threaded_pg(self, test_name, rank, world_size):
        """
        Run the current test associated with `test_name` using the threaded process group.
        """
        c10d.init_process_group(backend='threaded', rank=rank, world_size=world_size, store=self.__class__.global_store)
        self.perThreadSetUp()
        try:
            getattr(self, test_name)()
        except BaseException as ex:
            self.exception_queue.put((rank, sys.exc_info()))
            ProcessLocalGroup.exception_handle(ex)
        finally:
            c10d.destroy_process_group()
            self.perThreadTearDown()

    @classmethod
    def _join_threads(cls, threads, fn):
        timeout = TIMEOUT_DEFAULT
        try:
            for idx, thread in enumerate(threads):
                thread.join(max(0, timeout))
                if thread.is_alive():
                    MultiThreadedTestCase.exception_queue.put((idx, (TimeoutError, TimeoutError(f'Rank failed to join in under {timeout} seconds'), None)))
            ProcessLocalGroup.reset()
            failed_ranks = []
            while not cls.exception_queue.empty():
                failure = cls.exception_queue.get()
                failed_ranks.append(failure)
        finally:
            _uninstall_threaded_pg()
        cls._check_return_codes(failed_ranks, timeout, fn)

    @classmethod
    def _check_return_codes(cls, failed_ranks, timeout, fn):
        error_msg = ''
        skip_code = -1
        for rank, exc_info in failed_ranks:
            exc = exc_info[1]
            if isinstance(exc, unittest.SkipTest):
                logger.info('Thread %s skipping test %s for following reason: %s', rank, fn, str(exc))
                if skip_code < 0:
                    skip_code = TEST_SKIPS['generic'].exit_code
            elif isinstance(exc, TimeoutError):
                msg = f'Thread {rank} terminated or timed out after {timeout} seconds\n'
                logger.error(msg)
                raise RuntimeError(msg)
            elif isinstance(exc, Exception):
                msg = ''.join(traceback.format_exception(*exc_info))
                logger.error('Caught exception: \n%s exiting thread %s', msg, rank)
                error_msg += f'Thread {rank} exited with exception:\n{msg}\n'
            elif isinstance(exc, SystemExit):
                if type(exc.code) == int and skip_code < 0:
                    skip_code = exc.code
        if len(error_msg) > 0:
            raise RuntimeError(error_msg)
        if skip_code > 0:
            for skip in TEST_SKIPS.values():
                if skip_code == skip.exit_code:
                    if IS_SANDCASTLE:
                        logger.info('Skipping %s on sandcastle for the following reason: %s', fn, skip.message)
                        return
                    else:
                        raise unittest.SkipTest(skip.message)

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE

    @property
    def _current_test_name(self) -> str:
        return self.id().split('.')[-1]

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        """
        The reason why we have this util function instead of
        self.assertEqual is all threads are sharing one CPU RNG
        so the assertion result is only reliable on rank 0
        """
        if self.rank == rank:
            self.assertEqual(x, y, msg)

    def assertNotEqualOnRank(self, x, y, msg=None, *, rank=0):
        if self.rank == rank:
            self.assertNotEqual(x, y)