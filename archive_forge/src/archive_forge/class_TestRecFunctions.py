import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
class TestRecFunctions:

    def setup_method(self):
        x = np.array([1, 2])
        y = np.array([10, 20, 30])
        z = np.array([('A', 1.0), ('B', 2.0)], dtype=[('A', '|S3'), ('B', float)])
        w = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        self.data = (w, x, y, z)

    def test_zip_descr(self):
        w, x, y, z = self.data
        test = zip_descr((x, x), flatten=True)
        assert_equal(test, np.dtype([('', int), ('', int)]))
        test = zip_descr((x, x), flatten=False)
        assert_equal(test, np.dtype([('', int), ('', int)]))
        test = zip_descr((x, z), flatten=True)
        assert_equal(test, np.dtype([('', int), ('A', '|S3'), ('B', float)]))
        test = zip_descr((x, z), flatten=False)
        assert_equal(test, np.dtype([('', int), ('', [('A', '|S3'), ('B', float)])]))
        test = zip_descr((x, w), flatten=True)
        assert_equal(test, np.dtype([('', int), ('a', int), ('ba', float), ('bb', int)]))
        test = zip_descr((x, w), flatten=False)
        assert_equal(test, np.dtype([('', int), ('', [('a', int), ('b', [('ba', float), ('bb', int)])])]))

    def test_drop_fields(self):
        a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
        test = drop_fields(a, 'a')
        control = np.array([((2, 3.0),), ((5, 6.0),)], dtype=[('b', [('ba', float), ('bb', int)])])
        assert_equal(test, control)
        test = drop_fields(a, 'b')
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)
        test = drop_fields(a, ['ba'])
        control = np.array([(1, (3.0,)), (4, (6.0,))], dtype=[('a', int), ('b', [('bb', int)])])
        assert_equal(test, control)
        test = drop_fields(a, ['ba', 'bb'])
        control = np.array([(1,), (4,)], dtype=[('a', int)])
        assert_equal(test, control)
        test = drop_fields(a, ['a', 'b'])
        control = np.array([(), ()], dtype=[])
        assert_equal(test, control)

    def test_rename_fields(self):
        a = np.array([(1, (2, [3.0, 30.0])), (4, (5, [6.0, 60.0]))], dtype=[('a', int), ('b', [('ba', float), ('bb', (float, 2))])])
        test = rename_fields(a, {'a': 'A', 'bb': 'BB'})
        newdtype = [('A', int), ('b', [('ba', float), ('BB', (float, 2))])]
        control = a.view(newdtype)
        assert_equal(test.dtype, newdtype)
        assert_equal(test, control)

    def test_get_names(self):
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_names(ndtype)
        assert_equal(test, ('A', 'B'))
        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ('ba', 'bb'))))
        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names(ndtype)
        assert_equal(test, ('a', ('b', ())))
        ndtype = np.dtype([])
        test = get_names(ndtype)
        assert_equal(test, ())

    def test_get_names_flat(self):
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_names_flat(ndtype)
        assert_equal(test, ('A', 'B'))
        ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b', 'ba', 'bb'))
        ndtype = np.dtype([('a', int), ('b', [])])
        test = get_names_flat(ndtype)
        assert_equal(test, ('a', 'b'))
        ndtype = np.dtype([])
        test = get_names_flat(ndtype)
        assert_equal(test, ())

    def test_get_fieldstructure(self):
        ndtype = np.dtype([('A', '|S3'), ('B', float)])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {'A': [], 'B': []})
        ndtype = np.dtype([('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B']})
        ndtype = np.dtype([('A', int), ('B', [('BA', int), ('BB', [('BBA', int), ('BBB', int)])])])
        test = get_fieldstructure(ndtype)
        control = {'A': [], 'B': [], 'BA': ['B'], 'BB': ['B'], 'BBA': ['B', 'BB'], 'BBB': ['B', 'BB']}
        assert_equal(test, control)
        ndtype = np.dtype([])
        test = get_fieldstructure(ndtype)
        assert_equal(test, {})

    def test_find_duplicates(self):
        a = ma.array([(2, (2.0, 'B')), (1, (2.0, 'B')), (2, (2.0, 'B')), (1, (1.0, 'B')), (2, (2.0, 'B')), (2, (2.0, 'C'))], mask=[(0, (0, 0)), (0, (0, 0)), (0, (0, 0)), (0, (0, 0)), (1, (0, 0)), (0, (1, 0))], dtype=[('A', int), ('B', [('BA', float), ('BB', '|S1')])])
        test = find_duplicates(a, ignoremask=False, return_index=True)
        control = [0, 2]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])
        test = find_duplicates(a, key='A', return_index=True)
        control = [0, 1, 2, 3, 5]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])
        test = find_duplicates(a, key='B', return_index=True)
        control = [0, 1, 2, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])
        test = find_duplicates(a, key='BA', return_index=True)
        control = [0, 1, 2, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])
        test = find_duplicates(a, key='BB', return_index=True)
        control = [0, 1, 2, 3, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

    def test_find_duplicates_ignoremask(self):
        ndtype = [('a', int)]
        a = ma.array([1, 1, 1, 2, 2, 3, 3], mask=[0, 0, 1, 0, 0, 0, 1]).view(ndtype)
        test = find_duplicates(a, ignoremask=True, return_index=True)
        control = [0, 1, 3, 4]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])
        test = find_duplicates(a, ignoremask=False, return_index=True)
        control = [0, 1, 2, 3, 4, 6]
        assert_equal(sorted(test[-1]), control)
        assert_equal(test[0], a[test[-1]])

    def test_repack_fields(self):
        dt = np.dtype('u1,f4,i8', align=True)
        a = np.zeros(2, dtype=dt)
        assert_equal(repack_fields(dt), np.dtype('u1,f4,i8'))
        assert_equal(repack_fields(a).itemsize, 13)
        assert_equal(repack_fields(repack_fields(dt), align=True), dt)
        dt = np.dtype((np.record, dt))
        assert_(repack_fields(dt).type is np.record)

    def test_structured_to_unstructured(self, tmp_path):
        a = np.zeros(4, dtype=[('a', 'i4'), ('b', 'f4,u2'), ('c', 'f4', 2)])
        out = structured_to_unstructured(a)
        assert_equal(out, np.zeros((4, 5), dtype='f8'))
        b = np.array([(1, 2, 5), (4, 5, 7), (7, 8, 11), (10, 11, 12)], dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
        out = np.mean(structured_to_unstructured(b[['x', 'z']]), axis=-1)
        assert_equal(out, np.array([3.0, 5.5, 9.0, 11.0]))
        out = np.mean(structured_to_unstructured(b[['x']]), axis=-1)
        assert_equal(out, np.array([1.0, 4.0, 7.0, 10.0]))
        c = np.arange(20).reshape((4, 5))
        out = unstructured_to_structured(c, a.dtype)
        want = np.array([(0, (1.0, 2), [3.0, 4.0]), (5, (6.0, 7), [8.0, 9.0]), (10, (11.0, 12), [13.0, 14.0]), (15, (16.0, 17), [18.0, 19.0])], dtype=[('a', 'i4'), ('b', [('f0', 'f4'), ('f1', 'u2')]), ('c', 'f4', (2,))])
        assert_equal(out, want)
        d = np.array([(1, 2, 5), (4, 5, 7), (7, 8, 11), (10, 11, 12)], dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'f8')])
        assert_equal(apply_along_fields(np.mean, d), np.array([8.0 / 3, 16.0 / 3, 26.0 / 3, 11.0]))
        assert_equal(apply_along_fields(np.mean, d[['x', 'z']]), np.array([3.0, 5.5, 9.0, 11.0]))
        d = np.array([(1, 2, 5), (4, 5, 7), (7, 8, 11), (10, 11, 12)], dtype=[('x', 'i4'), ('y', 'i4'), ('z', 'i4')])
        dd = structured_to_unstructured(d)
        ddd = unstructured_to_structured(dd, d.dtype)
        assert_(np.shares_memory(dd, d))
        assert_(np.shares_memory(ddd, d))
        dd_attrib_rev = structured_to_unstructured(d[['z', 'x']])
        assert_equal(dd_attrib_rev, [[5, 1], [7, 4], [11, 7], [12, 10]])
        assert_(np.shares_memory(dd_attrib_rev, d))
        d = np.array([(1, [2, 3], [[4, 5], [6, 7]]), (8, [9, 10], [[11, 12], [13, 14]])], dtype=[('x0', 'i4'), ('x1', ('i4', 2)), ('x2', ('i4', (2, 2)))])
        dd = structured_to_unstructured(d)
        ddd = unstructured_to_structured(dd, d.dtype)
        assert_(np.shares_memory(dd, d))
        assert_(np.shares_memory(ddd, d))
        d_rev = d[::-1]
        dd_rev = structured_to_unstructured(d_rev)
        assert_equal(dd_rev, [[8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 5, 6, 7]])
        d_attrib_rev = d[['x2', 'x1', 'x0']]
        dd_attrib_rev = structured_to_unstructured(d_attrib_rev)
        assert_equal(dd_attrib_rev, [[4, 5, 6, 7, 2, 3, 1], [11, 12, 13, 14, 9, 10, 8]])
        d = np.array([(1, [2, 3], [[4, 5], [6, 7]], 32), (8, [9, 10], [[11, 12], [13, 14]], 64)], dtype=[('x0', 'i4'), ('x1', ('i4', 2)), ('x2', ('i4', (2, 2))), ('ignored', 'u1')])
        dd = structured_to_unstructured(d[['x0', 'x1', 'x2']])
        assert_(np.shares_memory(dd, d))
        assert_equal(dd, [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
        point = np.dtype([('x', int), ('y', int)])
        triangle = np.dtype([('a', point), ('b', point), ('c', point)])
        arr = np.zeros(10, triangle)
        res = structured_to_unstructured(arr, dtype=int)
        assert_equal(res, np.zeros((10, 6), dtype=int))

        def subarray(dt, shape):
            return np.dtype((dt, shape))

        def structured(*dts):
            return np.dtype([('x{}'.format(i), dt) for i, dt in enumerate(dts)])

        def inspect(dt, dtype=None):
            arr = np.zeros((), dt)
            ret = structured_to_unstructured(arr, dtype=dtype)
            backarr = unstructured_to_structured(ret, dt)
            return (ret.shape, ret.dtype, backarr.dtype)
        dt = structured(subarray(structured(np.int32, np.int32), 3))
        assert_equal(inspect(dt), ((6,), np.int32, dt))
        dt = structured(subarray(subarray(np.int32, 2), 2))
        assert_equal(inspect(dt), ((4,), np.int32, dt))
        dt = structured(np.int32)
        assert_equal(inspect(dt), ((1,), np.int32, dt))
        dt = structured(np.int32, subarray(subarray(np.int32, 2), 2))
        assert_equal(inspect(dt), ((5,), np.int32, dt))
        dt = structured()
        assert_raises(ValueError, structured_to_unstructured, np.zeros(3, dt))
        assert_raises(NotImplementedError, structured_to_unstructured, np.zeros(3, dt), dtype=np.int32)
        assert_raises(NotImplementedError, unstructured_to_structured, np.zeros((3, 0), dtype=np.int32))
        d_plain = np.array([(1, 2), (3, 4)], dtype=[('a', 'i4'), ('b', 'i4')])
        dd_expected = structured_to_unstructured(d_plain, copy=True)
        d = d_plain.view(np.recarray)
        dd = structured_to_unstructured(d, copy=False)
        ddd = structured_to_unstructured(d, copy=True)
        assert_(np.shares_memory(d, dd))
        assert_(type(dd) is np.recarray)
        assert_(type(ddd) is np.recarray)
        assert_equal(dd, dd_expected)
        assert_equal(ddd, dd_expected)
        d = np.memmap(tmp_path / 'memmap', mode='w+', dtype=d_plain.dtype, shape=d_plain.shape)
        d[:] = d_plain
        dd = structured_to_unstructured(d, copy=False)
        ddd = structured_to_unstructured(d, copy=True)
        assert_(np.shares_memory(d, dd))
        assert_(type(dd) is np.memmap)
        assert_(type(ddd) is np.memmap)
        assert_equal(dd, dd_expected)
        assert_equal(ddd, dd_expected)

    def test_unstructured_to_structured(self):
        a = np.zeros((20, 2))
        test_dtype_args = [('x', float), ('y', float)]
        test_dtype = np.dtype(test_dtype_args)
        field1 = unstructured_to_structured(a, dtype=test_dtype_args)
        field2 = unstructured_to_structured(a, dtype=test_dtype)
        assert_equal(field1, field2)

    def test_field_assignment_by_name(self):
        a = np.ones(2, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
        newdt = [('b', 'f4'), ('c', 'u1')]
        assert_equal(require_fields(a, newdt), np.ones(2, newdt))
        b = np.array([(1, 2), (3, 4)], dtype=newdt)
        assign_fields_by_name(a, b, zero_unassigned=False)
        assert_equal(a, np.array([(1, 1, 2), (1, 3, 4)], dtype=a.dtype))
        assign_fields_by_name(a, b)
        assert_equal(a, np.array([(0, 1, 2), (0, 3, 4)], dtype=a.dtype))
        a = np.ones(2, dtype=[('a', [('b', 'f8'), ('c', 'u1')])])
        newdt = [('a', [('c', 'u1')])]
        assert_equal(require_fields(a, newdt), np.ones(2, newdt))
        b = np.array([((2,),), ((3,),)], dtype=newdt)
        assign_fields_by_name(a, b, zero_unassigned=False)
        assert_equal(a, np.array([((1, 2),), ((1, 3),)], dtype=a.dtype))
        assign_fields_by_name(a, b)
        assert_equal(a, np.array([((0, 2),), ((0, 3),)], dtype=a.dtype))
        a, b = (np.array(3), np.array(0))
        assign_fields_by_name(b, a)
        assert_equal(b[()], 3)