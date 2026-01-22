import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
@unittest.skipUnless(HAS_SHMEM, 'requires multiprocess.shared_memory')
@hashlib_helper.requires_hashdigest('md5')
class _TestSharedMemory(BaseTestCase):
    ALLOWED_TYPES = ('processes',)

    @staticmethod
    def _attach_existing_shmem_then_write(shmem_name_or_obj, binary_data):
        if isinstance(shmem_name_or_obj, str):
            local_sms = shared_memory.SharedMemory(shmem_name_or_obj)
        else:
            local_sms = shmem_name_or_obj
        local_sms.buf[:len(binary_data)] = binary_data
        local_sms.close()

    def _new_shm_name(self, prefix):
        return prefix + str(os.getpid())

    def test_shared_memory_basics(self):
        name_tsmb = self._new_shm_name('test01_tsmb')
        sms = shared_memory.SharedMemory(name_tsmb, create=True, size=512)
        self.addCleanup(sms.unlink)
        self.assertEqual(sms.name, name_tsmb)
        self.assertGreaterEqual(sms.size, 512)
        self.assertGreaterEqual(len(sms.buf), sms.size)
        self.assertIn(sms.name, str(sms))
        self.assertIn(str(sms.size), str(sms))
        sms.buf[0] = 42
        self.assertEqual(sms.buf[0], 42)
        also_sms = shared_memory.SharedMemory(name_tsmb)
        self.assertEqual(also_sms.buf[0], 42)
        also_sms.close()
        same_sms = shared_memory.SharedMemory(name_tsmb, size=20 * sms.size)
        self.assertLess(same_sms.size, 20 * sms.size)
        same_sms.close()
        with self.assertRaises(ValueError):
            shared_memory.SharedMemory(create=True, size=-2)
        with self.assertRaises(ValueError):
            shared_memory.SharedMemory(create=False)
        with unittest.mock.patch('multiprocess.shared_memory._make_filename') as mock_make_filename:
            NAME_PREFIX = shared_memory._SHM_NAME_PREFIX
            names = [self._new_shm_name('test01_fn'), self._new_shm_name('test02_fn')]
            names = [NAME_PREFIX + name for name in names]
            mock_make_filename.side_effect = names
            shm1 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm1.unlink)
            self.assertEqual(shm1._name, names[0])
            mock_make_filename.side_effect = names
            shm2 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm2.unlink)
            self.assertEqual(shm2._name, names[1])
        if shared_memory._USE_POSIX:
            name_dblunlink = self._new_shm_name('test01_dblunlink')
            sms_uno = shared_memory.SharedMemory(name_dblunlink, create=True, size=5000)
            with self.assertRaises(FileNotFoundError):
                try:
                    self.assertGreaterEqual(sms_uno.size, 5000)
                    sms_duo = shared_memory.SharedMemory(name_dblunlink)
                    sms_duo.unlink()
                    sms_duo.close()
                    sms_uno.close()
                finally:
                    sms_uno.unlink()
        with self.assertRaises(FileExistsError):
            there_can_only_be_one_sms = shared_memory.SharedMemory(name_tsmb, create=True, size=512)
        if shared_memory._USE_POSIX:

            class OptionalAttachSharedMemory(shared_memory.SharedMemory):
                _flags = os.O_CREAT | os.O_RDWR
            ok_if_exists_sms = OptionalAttachSharedMemory(name_tsmb)
            self.assertEqual(ok_if_exists_sms.size, sms.size)
            ok_if_exists_sms.close()
        with self.assertRaises(FileNotFoundError):
            nonexisting_sms = shared_memory.SharedMemory('test01_notthere')
            nonexisting_sms.unlink()
        sms.close()

    @unittest.skipIf(True, 'fails with dill >= 0.3.5')
    def test_shared_memory_recreate(self):
        with unittest.mock.patch('multiprocess.shared_memory._make_filename') as mock_make_filename:
            NAME_PREFIX = shared_memory._SHM_NAME_PREFIX
            names = [self._new_shm_name('test03_fn'), self._new_shm_name('test04_fn')]
            names = [NAME_PREFIX + name for name in names]
            mock_make_filename.side_effect = names
            shm1 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm1.unlink)
            self.assertEqual(shm1._name, names[0])
            mock_make_filename.side_effect = names
            shm2 = shared_memory.SharedMemory(create=True, size=1)
            self.addCleanup(shm2.unlink)
            self.assertEqual(shm2._name, names[1])

    def test_invalid_shared_memory_cration(self):
        with self.assertRaises(ValueError):
            sms_invalid = shared_memory.SharedMemory(create=True, size=-1)
        with self.assertRaises(ValueError):
            sms_invalid = shared_memory.SharedMemory(create=True, size=0)
        with self.assertRaises(ValueError):
            sms_invalid = shared_memory.SharedMemory(create=True)

    def test_shared_memory_pickle_unpickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sms = shared_memory.SharedMemory(create=True, size=512)
                self.addCleanup(sms.unlink)
                sms.buf[0:6] = b'pickle'
                pickled_sms = pickle.dumps(sms, protocol=proto)
                sms2 = pickle.loads(pickled_sms)
                self.assertIsInstance(sms2, shared_memory.SharedMemory)
                self.assertEqual(sms.name, sms2.name)
                self.assertEqual(bytes(sms.buf[0:6]), b'pickle')
                self.assertEqual(bytes(sms2.buf[0:6]), b'pickle')
                sms.buf[0:6] = b'newval'
                self.assertEqual(bytes(sms.buf[0:6]), b'newval')
                self.assertEqual(bytes(sms2.buf[0:6]), b'newval')
                sms2.buf[0:6] = b'oldval'
                self.assertEqual(bytes(sms.buf[0:6]), b'oldval')
                self.assertEqual(bytes(sms2.buf[0:6]), b'oldval')

    def test_shared_memory_pickle_unpickle_dead_object(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sms = shared_memory.SharedMemory(create=True, size=512)
                sms.buf[0:6] = b'pickle'
                pickled_sms = pickle.dumps(sms, protocol=proto)
                sms.close()
                sms.unlink()
                with self.assertRaises(FileNotFoundError):
                    pickle.loads(pickled_sms)

    def test_shared_memory_across_processes(self):
        sms = shared_memory.SharedMemory(create=True, size=512)
        self.addCleanup(sms.unlink)
        p = self.Process(target=self._attach_existing_shmem_then_write, args=(sms.name, b'howdy'))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(bytes(sms.buf[:5]), b'howdy')
        p = self.Process(target=self._attach_existing_shmem_then_write, args=(sms, b'HELLO'))
        p.daemon = True
        p.start()
        p.join()
        self.assertEqual(bytes(sms.buf[:5]), b'HELLO')
        sms.close()

    @unittest.skipIf(os.name != 'posix', 'not feasible in non-posix platforms')
    def test_shared_memory_SharedMemoryServer_ignores_sigint(self):
        smm = multiprocessing.managers.SharedMemoryManager()
        smm.start()
        sl = smm.ShareableList(range(10))
        os.kill(smm._process.pid, signal.SIGINT)
        sl2 = smm.ShareableList(range(10))
        with self.assertRaises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
        smm.shutdown()

    @unittest.skipIf(os.name != 'posix', 'resource_tracker is posix only')
    def test_shared_memory_SharedMemoryManager_reuses_resource_tracker(self):
        cmd = 'if 1:\n            from multiprocessing.managers import SharedMemoryManager\n\n\n            smm = SharedMemoryManager()\n            smm.start()\n            sl = smm.ShareableList(range(10))\n            smm.shutdown()\n        '
        rc, out, err = test.support.script_helper.assert_python_ok('-c', cmd, **ENV)
        self.assertFalse(err)

    def test_shared_memory_SharedMemoryManager_basics(self):
        smm1 = multiprocessing.managers.SharedMemoryManager()
        with self.assertRaises(ValueError):
            smm1.SharedMemory(size=9)
        smm1.start()
        lol = [smm1.ShareableList(range(i)) for i in range(5, 10)]
        lom = [smm1.SharedMemory(size=j) for j in range(32, 128, 16)]
        doppleganger_list0 = shared_memory.ShareableList(name=lol[0].shm.name)
        self.assertEqual(len(doppleganger_list0), 5)
        doppleganger_shm0 = shared_memory.SharedMemory(name=lom[0].name)
        self.assertGreaterEqual(len(doppleganger_shm0.buf), 32)
        held_name = lom[0].name
        smm1.shutdown()
        if sys.platform != 'win32':
            with self.assertRaises(FileNotFoundError):
                absent_shm = shared_memory.SharedMemory(name=held_name)
        with multiprocessing.managers.SharedMemoryManager() as smm2:
            sl = smm2.ShareableList('howdy')
            shm = smm2.SharedMemory(size=128)
            held_name = sl.shm.name
        if sys.platform != 'win32':
            with self.assertRaises(FileNotFoundError):
                absent_sl = shared_memory.ShareableList(name=held_name)

    def test_shared_memory_ShareableList_basics(self):
        sl = shared_memory.ShareableList(['howdy', b'HoWdY', -273.154, 100, None, True, 42])
        self.addCleanup(sl.shm.unlink)
        self.assertIn(sl.shm.name, str(sl))
        self.assertIn(str(list(sl)), str(sl))
        with self.assertRaises(IndexError):
            sl[7]
        with self.assertRaises(IndexError):
            sl[7] = 2
        current_format = sl._get_packing_format(0)
        sl[0] = 'howdy'
        self.assertEqual(current_format, sl._get_packing_format(0))
        self.assertEqual(sl.format, '8s8sdqxxxxxx?xxxxxxxx?q')
        self.assertEqual(len(sl), 7)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(ValueError):
                sl.index('100')
            self.assertEqual(sl.index(100), 3)
        self.assertEqual(sl[0], 'howdy')
        self.assertEqual(sl[-2], True)
        self.assertEqual(tuple(sl), ('howdy', b'HoWdY', -273.154, 100, None, True, 42))
        sl[3] = 42
        self.assertEqual(sl[3], 42)
        sl[4] = 'some'
        self.assertEqual(sl[4], 'some')
        self.assertEqual(sl.format, '8s8sdq8sxxxxxxx?q')
        with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
            sl[4] = 'far too many'
        self.assertEqual(sl[4], 'some')
        sl[0] = 'encodés'
        self.assertEqual(sl[0], 'encodés')
        self.assertEqual(sl[1], b'HoWdY')
        with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
            sl[0] = 'encodées'
        self.assertEqual(sl[1], b'HoWdY')
        with self.assertRaisesRegex(ValueError, 'exceeds available storage'):
            sl[1] = b'123456789'
        self.assertEqual(sl[1], b'HoWdY')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(sl.count(42), 2)
            self.assertEqual(sl.count(b'HoWdY'), 1)
            self.assertEqual(sl.count(b'adios'), 0)
        name_duplicate = self._new_shm_name('test03_duplicate')
        sl_copy = shared_memory.ShareableList(sl, name=name_duplicate)
        try:
            self.assertNotEqual(sl.shm.name, sl_copy.shm.name)
            self.assertEqual(name_duplicate, sl_copy.shm.name)
            self.assertEqual(list(sl), list(sl_copy))
            self.assertEqual(sl.format, sl_copy.format)
            sl_copy[-1] = 77
            self.assertEqual(sl_copy[-1], 77)
            self.assertNotEqual(sl[-1], 77)
            sl_copy.shm.close()
        finally:
            sl_copy.shm.unlink()
        sl_tethered = shared_memory.ShareableList(name=sl.shm.name)
        self.assertEqual(sl.shm.name, sl_tethered.shm.name)
        sl_tethered[-1] = 880
        self.assertEqual(sl[-1], 880)
        sl_tethered.shm.close()
        sl.shm.close()
        empty_sl = shared_memory.ShareableList()
        try:
            self.assertEqual(len(empty_sl), 0)
            self.assertEqual(empty_sl.format, '')
            self.assertEqual(empty_sl.count('any'), 0)
            with self.assertRaises(ValueError):
                empty_sl.index(None)
            empty_sl.shm.close()
        finally:
            empty_sl.shm.unlink()

    def test_shared_memory_ShareableList_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sl = shared_memory.ShareableList(range(10))
                self.addCleanup(sl.shm.unlink)
                serialized_sl = pickle.dumps(sl, protocol=proto)
                deserialized_sl = pickle.loads(serialized_sl)
                self.assertIsInstance(deserialized_sl, shared_memory.ShareableList)
                self.assertEqual(deserialized_sl[-1], 9)
                self.assertIsNot(sl, deserialized_sl)
                deserialized_sl[4] = 'changed'
                self.assertEqual(sl[4], 'changed')
                sl[3] = 'newvalue'
                self.assertEqual(deserialized_sl[3], 'newvalue')
                larger_sl = shared_memory.ShareableList(range(400))
                self.addCleanup(larger_sl.shm.unlink)
                serialized_larger_sl = pickle.dumps(larger_sl, protocol=proto)
                self.assertEqual(len(serialized_sl), len(serialized_larger_sl))
                larger_sl.shm.close()
                deserialized_sl.shm.close()
                sl.shm.close()

    def test_shared_memory_ShareableList_pickling_dead_object(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                sl = shared_memory.ShareableList(range(10))
                serialized_sl = pickle.dumps(sl, protocol=proto)
                sl.shm.close()
                sl.shm.unlink()
                with self.assertRaises(FileNotFoundError):
                    pickle.loads(serialized_sl)

    def test_shared_memory_cleaned_after_process_termination(self):
        cmd = "if 1:\n            import os, time, sys\n            from multiprocessing import shared_memory\n\n            # Create a shared_memory segment, and send the segment name\n            sm = shared_memory.SharedMemory(create=True, size=10)\n            sys.stdout.write(sm.name + '\\n')\n            sys.stdout.flush()\n            time.sleep(100)\n        "
        with subprocess.Popen([sys.executable, '-E', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            name = p.stdout.readline().strip().decode()
            p.terminate()
            p.wait()
            deadline = time.monotonic() + support.LONG_TIMEOUT
            t = 0.1
            while time.monotonic() < deadline:
                time.sleep(t)
                t = min(t * 2, 5)
                try:
                    smm = shared_memory.SharedMemory(name, create=False)
                except FileNotFoundError:
                    break
            else:
                raise AssertionError('A SharedMemory segment was leaked after a process was abruptly terminated.')
            if os.name == 'posix':
                resource_tracker.unregister(f'/{name}', 'shared_memory')
                err = p.stderr.read().decode()
                self.assertIn('resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown', err)