import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
class TestFingerprint(TestCase):
    """
    Tests for _dispatcher.compute_fingerprint()

    Each fingerprint must denote values of only one Numba type (this is
    the condition for correctness), but values of a Numba type may be
    denoted by several distinct fingerprints (it only makes the cache
    less efficient).
    """

    def test_floats(self):
        s = compute_fingerprint(1.0)
        self.assertEqual(compute_fingerprint(2.0), s)
        s = compute_fingerprint(np.float32())
        self.assertEqual(compute_fingerprint(np.float32(2.0)), s)
        self.assertNotEqual(compute_fingerprint(np.float64()), s)

    def test_ints(self):
        s = compute_fingerprint(1)
        for v in (-1, 2 ** 60):
            self.assertEqual(compute_fingerprint(v), s)
        distinct = set()
        for tp in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
            tp = getattr(np, tp)
            distinct.add(compute_fingerprint(tp()))
        self.assertEqual(len(distinct), 8, distinct)

    def test_bool(self):
        s = compute_fingerprint(True)
        self.assertEqual(compute_fingerprint(False), s)
        self.assertNotEqual(compute_fingerprint(1), s)

    def test_complex(self):
        s = compute_fingerprint(1j)
        self.assertEqual(s, compute_fingerprint(1 + 0j))
        s = compute_fingerprint(np.complex64())
        self.assertEqual(compute_fingerprint(np.complex64(2.0)), s)
        self.assertNotEqual(compute_fingerprint(np.complex128()), s)

    def test_none(self):
        compute_fingerprint(None)

    def test_enums(self):
        with self.assertRaises(NotImplementedError):
            compute_fingerprint(Color.red)
        with self.assertRaises(NotImplementedError):
            compute_fingerprint(RequestError.not_found)

    def test_records(self):
        d1 = np.dtype([('m', np.int32), ('n', np.int64)])
        d2 = np.dtype([('m', np.int32), ('n', np.int16)])
        v1 = np.empty(1, dtype=d1)[0]
        v2 = np.empty(1, dtype=d2)[0]
        self.assertNotEqual(compute_fingerprint(v1), compute_fingerprint(v2))

    def test_datetime(self):
        a = np.datetime64(1, 'Y')
        b = np.datetime64(2, 'Y')
        c = np.datetime64(2, 's')
        d = np.timedelta64(2, 's')
        self.assertEqual(compute_fingerprint(a), compute_fingerprint(b))
        distinct = set((compute_fingerprint(x) for x in (a, c, d)))
        self.assertEqual(len(distinct), 3, distinct)

    def test_arrays(self):
        distinct = DistinctChecker()
        arr = np.empty(4, dtype=np.float64)
        s = compute_fingerprint(arr)
        distinct.add(s)
        self.assertEqual(compute_fingerprint(arr[:1]), s)
        distinct.add(compute_fingerprint(arr[::2]))
        distinct.add(compute_fingerprint(arr.astype(np.complex64)))
        arr.setflags(write=False)
        distinct.add(compute_fingerprint(arr))
        arr = np.empty((4, 4), dtype=np.float64)
        distinct.add(compute_fingerprint(arr))
        distinct.add(compute_fingerprint(arr.T))
        distinct.add(compute_fingerprint(arr[::2]))
        arr = np.empty((), dtype=np.float64)
        distinct.add(compute_fingerprint(arr))
        arr = np.empty(5, dtype=recordtype)
        s = compute_fingerprint(arr)
        distinct.add(s)
        self.assertEqual(compute_fingerprint(arr[:1]), s)
        arr = np.empty(5, dtype=recordtype2)
        distinct.add(compute_fingerprint(arr))
        arr = np.empty(5, dtype=recordtype3)
        distinct.add(compute_fingerprint(arr))
        a = np.recarray(1, dtype=recordtype)
        b = np.recarray(1, dtype=recordtype)
        self.assertEqual(compute_fingerprint(a), compute_fingerprint(b))

    def test_buffers(self):
        distinct = DistinctChecker()
        s = compute_fingerprint(b'')
        self.assertEqual(compute_fingerprint(b'xx'), s)
        distinct.add(s)
        distinct.add(compute_fingerprint(bytearray()))
        distinct.add(compute_fingerprint(memoryview(b'')))
        m_uint8_1d = compute_fingerprint(memoryview(bytearray()))
        distinct.add(m_uint8_1d)
        arr = array.array('B', [42])
        distinct.add(compute_fingerprint(arr))
        self.assertEqual(compute_fingerprint(memoryview(arr)), m_uint8_1d)
        for array_code in 'bi':
            arr = array.array(array_code, [0, 1, 2])
            distinct.add(compute_fingerprint(arr))
            distinct.add(compute_fingerprint(memoryview(arr)))
        arr = np.empty(16, dtype=np.uint8)
        distinct.add(compute_fingerprint(arr))
        self.assertEqual(compute_fingerprint(memoryview(arr)), m_uint8_1d)
        arr = arr.reshape((4, 4))
        distinct.add(compute_fingerprint(arr))
        distinct.add(compute_fingerprint(memoryview(arr)))
        arr = arr.T
        distinct.add(compute_fingerprint(arr))
        distinct.add(compute_fingerprint(memoryview(arr)))
        arr = arr[::2]
        distinct.add(compute_fingerprint(arr))
        distinct.add(compute_fingerprint(memoryview(arr)))
        m = mmap.mmap(-1, 16384)
        distinct.add(compute_fingerprint(m))
        self.assertEqual(compute_fingerprint(memoryview(m)), m_uint8_1d)

    def test_dtype(self):
        distinct = DistinctChecker()
        s = compute_fingerprint(np.dtype('int64'))
        self.assertEqual(compute_fingerprint(np.dtype('int64')), s)
        distinct.add(s)
        for descr in ('int32', 'm8[s]', 'm8[W]', 'M8[s]'):
            distinct.add(np.dtype(descr))
        distinct.add(recordtype)
        distinct.add(recordtype2)
        a = np.recarray(1, dtype=recordtype)
        b = np.recarray(1, dtype=recordtype)
        self.assertEqual(compute_fingerprint(a.dtype), compute_fingerprint(b.dtype))

    def test_tuples(self):
        distinct = DistinctChecker()
        s = compute_fingerprint((1,))
        self.assertEqual(compute_fingerprint((2,)), s)
        distinct.add(s)
        distinct.add(compute_fingerprint(()))
        distinct.add(compute_fingerprint((1, 2, 3)))
        distinct.add(compute_fingerprint((1j, 2, 3)))
        distinct.add(compute_fingerprint((1, (), np.empty(5))))
        distinct.add(compute_fingerprint((1, (), np.empty((5, 1)))))

    def test_lists(self):
        distinct = DistinctChecker()
        s = compute_fingerprint([1])
        self.assertEqual(compute_fingerprint([2, 3]), s)
        distinct.add(s)
        distinct.add(compute_fingerprint([1j]))
        distinct.add(compute_fingerprint([4.5, 6.7]))
        distinct.add(compute_fingerprint([(1,)]))
        with self.assertRaises(ValueError):
            compute_fingerprint([])

    def test_sets(self):
        distinct = DistinctChecker()
        s = compute_fingerprint(set([1]))
        self.assertEqual(compute_fingerprint(set([2, 3])), s)
        distinct.add(s)
        distinct.add(compute_fingerprint([1]))
        distinct.add(compute_fingerprint(set([1j])))
        distinct.add(compute_fingerprint(set([4.5, 6.7])))
        distinct.add(compute_fingerprint(set([(1,)])))
        with self.assertRaises(ValueError):
            compute_fingerprint(set())
        with self.assertRaises(NotImplementedError):
            compute_fingerprint(frozenset([2, 3]))

    def test_omitted_args(self):
        distinct = DistinctChecker()
        v0 = OmittedArg(0.0)
        v1 = OmittedArg(1.0)
        v2 = OmittedArg(1)
        s = compute_fingerprint(v0)
        self.assertEqual(compute_fingerprint(v1), s)
        distinct.add(s)
        distinct.add(compute_fingerprint(v2))
        distinct.add(compute_fingerprint(0.0))
        distinct.add(compute_fingerprint(1))

    def test_complicated_type(self):
        t = None
        for i in range(1000):
            t = (t,)
        s = compute_fingerprint(t)