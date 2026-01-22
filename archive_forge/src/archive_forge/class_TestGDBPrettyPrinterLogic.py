from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
class TestGDBPrettyPrinterLogic(TestCase):

    def setUp(self):
        mock_modules = {'gdb': Mock(), 'gdb.printing': Mock()}
        self.patched_sys = patch.dict('sys.modules', mock_modules)
        self.patched_sys.start()
        import gdb

        class SelectedInferior:

            def read_memory(self, data, extent):
                buf = (ct.c_char * extent).from_address(data)
                return buf.raw
        si = SelectedInferior()
        gdb.configure_mock(**{'selected_inferior': lambda: si})

    def tearDown(self):
        self.patched_sys.stop()

    def get_gdb_repr(self, array):
        from numba.misc import gdb_print_extension

        class DISubrange:

            def __init__(self, lo, hi):
                self._lo = lo
                self._hi = hi

            @property
            def type(self):
                return self

            def range(self):
                return (self._lo, self._hi)

        class DW_TAG_array_type:

            def __init__(self, lo, hi):
                self._lo, self._hi = (lo, hi)

            def fields(self):
                return [DISubrange(self._lo, self._hi)]

        class DIDerivedType_tuple:

            def __init__(self, the_tuple):
                self._type = DW_TAG_array_type(0, len(the_tuple) - 1)
                self._tuple = the_tuple

            @property
            def type(self):
                return self._type

            def __getitem__(self, item):
                return self._tuple[item]

        class DICompositeType_Array:

            def __init__(self, arr, type_str):
                self._arr = arr
                self._type_str = type_str

            def __getitem__(self, item):
                return getattr(self, item)

            @property
            def data(self):
                return self._arr.ctypes.data

            @property
            def itemsize(self):
                return self._arr.itemsize

            @property
            def shape(self):
                return DIDerivedType_tuple(self._arr.shape)

            @property
            def strides(self):
                return DIDerivedType_tuple(self._arr.strides)

            @property
            def type(self):
                return self._type_str
        dmm = datamodel.default_manager
        array_model = datamodel.models.ArrayModel(dmm, typeof(array))
        data_type = array_model.get_data_type()
        type_str = f'{typeof(array)} ({data_type.structure_repr()})'
        fake_gdb_arr = DICompositeType_Array(array, type_str)
        printer = gdb_print_extension.NumbaArrayPrinter(fake_gdb_arr)
        return printer.to_string().strip()

    def check(self, array):
        gdb_printed = self.get_gdb_repr(array)
        self.assertEqual(str(gdb_printed), str(array))

    def test_np_array_printer_simple_numeric_types(self):
        n = 4
        m = 3
        for dt in (np.int8, np.uint16, np.int64, np.float32, np.complex128):
            arr = np.arange(m * n, dtype=dt).reshape(m, n)
            self.check(arr)

    def test_np_array_printer_simple_numeric_types_strided(self):
        n_tests = 30
        np.random.seed(0)
        for _ in range(n_tests):
            shape = np.random.randint(1, high=12, size=np.random.randint(1, 5))
            tmp = np.arange(np.prod(shape)).reshape(shape)
            slices = []
            for x in shape:
                start = np.random.randint(0, x)
                stop = np.random.randint(start + 1, max(start + 1, x + 3))
                step = np.random.randint(1, 3)
                strd = slice(start, stop, step)
                slices.append(strd)
            arr = tmp[tuple(slices)]
            self.check(arr)

    def test_np_array_printer_simple_structured_dtype(self):
        n = 4
        m = 3
        aligned = np.dtype([('x', np.int16), ('y', np.float64)], align=True)
        unaligned = np.dtype([('x', np.int16), ('y', np.float64)], align=False)
        for dt in (aligned, unaligned):
            arr = np.empty(m * n, dtype=dt).reshape(m, n)
            arr['x'] = np.arange(m * n, dtype=dt['x']).reshape(m, n)
            arr['y'] = 100 * np.arange(m * n, dtype=dt['y']).reshape(m, n)
            self.check(arr)

    def test_np_array_printer_chr_array(self):
        arr = np.array(['abcde'])
        self.check(arr)

    def test_np_array_printer_unichr_structured_dtype(self):
        n = 4
        m = 3
        dt = np.dtype([('x', '<U5'), ('y', np.float64)], align=True)
        arr = np.zeros(m * n, dtype=dt).reshape(m, n)
        rep = self.get_gdb_repr(arr)
        self.assertIn('array[Exception:', rep)
        self.assertIn('Unsupported sub-type', rep)
        self.assertIn('[unichr x 5]', rep)

    def test_np_array_printer_nested_array_structured_dtype(self):
        n = 4
        m = 3
        dt = np.dtype([('x', np.int16, (2,)), ('y', np.float64)], align=True)
        arr = np.zeros(m * n, dtype=dt).reshape(m, n)
        rep = self.get_gdb_repr(arr)
        self.assertIn('array[Exception:', rep)
        self.assertIn('Unsupported sub-type', rep)
        self.assertIn('nestedarray(int16', rep)