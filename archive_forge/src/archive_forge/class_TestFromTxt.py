import sys
import gc
import gzip
import os
import threading
import time
import warnings
import io
import re
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
from datetime import datetime
import locale
from multiprocessing import Value, get_context
from ctypes import c_bool
import numpy as np
import numpy.ma as ma
from numpy.lib._iotools import ConverterError, ConversionWarning
from numpy.compat import asbytes
from numpy.ma.testutils import assert_equal
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
class TestFromTxt(LoadTxtBase):
    loadfunc = staticmethod(np.genfromtxt)

    def test_record(self):
        data = TextIO('1 2\n3 4')
        test = np.genfromtxt(data, dtype=[('x', np.int32), ('y', np.int32)])
        control = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
        assert_equal(test, control)
        data = TextIO('M 64.0 75.0\nF 25.0 60.0')
        descriptor = {'names': ('gender', 'age', 'weight'), 'formats': ('S1', 'i4', 'f4')}
        control = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)], dtype=descriptor)
        test = np.genfromtxt(data, dtype=descriptor)
        assert_equal(test, control)

    def test_array(self):
        data = TextIO('1 2\n3 4')
        control = np.array([[1, 2], [3, 4]], dtype=int)
        test = np.genfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        data.seek(0)
        control = np.array([[1, 2], [3, 4]], dtype=float)
        test = np.loadtxt(data, dtype=float)
        assert_array_equal(test, control)

    def test_1D(self):
        control = np.array([1, 2, 3, 4], int)
        data = TextIO('1\n2\n3\n4\n')
        test = np.genfromtxt(data, dtype=int)
        assert_array_equal(test, control)
        data = TextIO('1,2,3,4\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',')
        assert_array_equal(test, control)

    def test_comments(self):
        control = np.array([1, 2, 3, 5], int)
        data = TextIO('# comment\n1,2,3,5\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',', comments='#')
        assert_equal(test, control)
        data = TextIO('1,2,3,5# comment\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',', comments='#')
        assert_equal(test, control)

    def test_skiprows(self):
        control = np.array([1, 2, 3, 5], int)
        kwargs = dict(dtype=int, delimiter=',')
        data = TextIO('comment\n1,2,3,5\n')
        test = np.genfromtxt(data, skip_header=1, **kwargs)
        assert_equal(test, control)
        data = TextIO('# comment\n1,2,3,5\n')
        test = np.loadtxt(data, skiprows=1, **kwargs)
        assert_equal(test, control)

    def test_skip_footer(self):
        data = ['# %i' % i for i in range(1, 6)]
        data.append('A, B, C')
        data.extend(['%i,%3.1f,%03s' % (i, i, i) for i in range(51)])
        data[-1] = '99,99'
        kwargs = dict(delimiter=',', names=True, skip_header=5, skip_footer=10)
        test = np.genfromtxt(TextIO('\n'.join(data)), **kwargs)
        ctrl = np.array([('%f' % i, '%f' % i, '%f' % i) for i in range(41)], dtype=[(_, float) for _ in 'ABC'])
        assert_equal(test, ctrl)

    def test_skip_footer_with_invalid(self):
        with suppress_warnings() as sup:
            sup.filter(ConversionWarning)
            basestr = '1 1\n2 2\n3 3\n4 4\n5  \n6  \n7  \n'
            assert_raises(ValueError, np.genfromtxt, TextIO(basestr), skip_footer=1)
            a = np.genfromtxt(TextIO(basestr), skip_footer=1, invalid_raise=False)
            assert_equal(a, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
            a = np.genfromtxt(TextIO(basestr), skip_footer=3)
            assert_equal(a, np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]))
            basestr = '1 1\n2  \n3 3\n4 4\n5  \n6 6\n7 7\n'
            a = np.genfromtxt(TextIO(basestr), skip_footer=1, invalid_raise=False)
            assert_equal(a, np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 4.0], [6.0, 6.0]]))
            a = np.genfromtxt(TextIO(basestr), skip_footer=3, invalid_raise=False)
            assert_equal(a, np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 4.0]]))

    def test_header(self):
        data = TextIO('gender age weight\nM 64.0 75.0\nF 25.0 60.0')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(data, dtype=None, names=True)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        control = {'gender': np.array([b'M', b'F']), 'age': np.array([64.0, 25.0]), 'weight': np.array([75.0, 60.0])}
        assert_equal(test['gender'], control['gender'])
        assert_equal(test['age'], control['age'])
        assert_equal(test['weight'], control['weight'])

    def test_auto_dtype(self):
        data = TextIO('A 64 75.0 3+4j True\nBCD 25 60.0 5+6j False')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(data, dtype=None)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        control = [np.array([b'A', b'BCD']), np.array([64, 25]), np.array([75.0, 60.0]), np.array([3 + 4j, 5 + 6j]), np.array([True, False])]
        assert_equal(test.dtype.names, ['f0', 'f1', 'f2', 'f3', 'f4'])
        for i, ctrl in enumerate(control):
            assert_equal(test['f%i' % i], ctrl)

    def test_auto_dtype_uniform(self):
        data = TextIO('1 2 3 4\n5 6 7 8\n')
        test = np.genfromtxt(data, dtype=None)
        control = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert_equal(test, control)

    def test_fancy_dtype(self):
        data = TextIO('1,2,3.0\n4,5,6.0\n')
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = np.genfromtxt(data, dtype=fancydtype, delimiter=',')
        control = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=fancydtype)
        assert_equal(test, control)

    def test_names_overwrite(self):
        descriptor = {'names': ('g', 'a', 'w'), 'formats': ('S1', 'i4', 'f4')}
        data = TextIO(b'M 64.0 75.0\nF 25.0 60.0')
        names = ('gender', 'age', 'weight')
        test = np.genfromtxt(data, dtype=descriptor, names=names)
        descriptor['names'] = names
        control = np.array([('M', 64.0, 75.0), ('F', 25.0, 60.0)], dtype=descriptor)
        assert_equal(test, control)

    def test_bad_fname(self):
        with pytest.raises(TypeError, match='fname must be a string,'):
            np.genfromtxt(123)

    def test_commented_header(self):
        data = TextIO('\n#gender age weight\nM   21  72.100000\nF   35  58.330000\nM   33  21.99\n        ')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(data, names=True, dtype=None)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        ctrl = np.array([('M', 21, 72.1), ('F', 35, 58.33), ('M', 33, 21.99)], dtype=[('gender', '|S1'), ('age', int), ('weight', float)])
        assert_equal(test, ctrl)
        data = TextIO(b'\n# gender age weight\nM   21  72.100000\nF   35  58.330000\nM   33  21.99\n        ')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(data, names=True, dtype=None)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        assert_equal(test, ctrl)

    def test_names_and_comments_none(self):
        data = TextIO('col1 col2\n 1 2\n 3 4')
        test = np.genfromtxt(data, dtype=(int, int), comments=None, names=True)
        control = np.array([(1, 2), (3, 4)], dtype=[('col1', int), ('col2', int)])
        assert_equal(test, control)

    def test_file_is_closed_on_error(self):
        with tempdir() as tmpdir:
            fpath = os.path.join(tmpdir, 'test.csv')
            with open(fpath, 'wb') as f:
                f.write('ϖ'.encode())
            with assert_no_warnings():
                with pytest.raises(UnicodeDecodeError):
                    np.genfromtxt(fpath, encoding='ascii')

    def test_autonames_and_usecols(self):
        data = TextIO('A B C D\n aaaa 121 45 9.1')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(data, usecols=('A', 'C', 'D'), names=True, dtype=None)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        control = np.array(('aaaa', 45, 9.1), dtype=[('A', '|S4'), ('C', int), ('D', float)])
        assert_equal(test, control)

    def test_converters_with_usecols(self):
        data = TextIO('1,2,3,,5\n6,7,8,9,10\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',', converters={3: lambda s: int(s or -999)}, usecols=(1, 3))
        control = np.array([[2, -999], [7, 9]], int)
        assert_equal(test, control)

    def test_converters_with_usecols_and_names(self):
        data = TextIO('A B C D\n aaaa 121 45 9.1')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(data, usecols=('A', 'C', 'D'), names=True, dtype=None, converters={'C': lambda s: 2 * int(s)})
            assert_(w[0].category is np.VisibleDeprecationWarning)
        control = np.array(('aaaa', 90, 9.1), dtype=[('A', '|S4'), ('C', int), ('D', float)])
        assert_equal(test, control)

    def test_converters_cornercases(self):
        converter = {'date': lambda s: strptime(s, '%Y-%m-%d %H:%M:%SZ')}
        data = TextIO('2009-02-03 12:00:00Z, 72214.0')
        test = np.genfromtxt(data, delimiter=',', dtype=None, names=['date', 'stid'], converters=converter)
        control = np.array((datetime(2009, 2, 3), 72214.0), dtype=[('date', np.object_), ('stid', float)])
        assert_equal(test, control)

    def test_converters_cornercases2(self):
        converter = {'date': lambda s: np.datetime64(strptime(s, '%Y-%m-%d %H:%M:%SZ'))}
        data = TextIO('2009-02-03 12:00:00Z, 72214.0')
        test = np.genfromtxt(data, delimiter=',', dtype=None, names=['date', 'stid'], converters=converter)
        control = np.array((datetime(2009, 2, 3), 72214.0), dtype=[('date', 'datetime64[us]'), ('stid', float)])
        assert_equal(test, control)

    def test_unused_converter(self):
        data = TextIO('1 21\n  3 42\n')
        test = np.genfromtxt(data, usecols=(1,), converters={0: lambda s: int(s, 16)})
        assert_equal(test, [21, 42])
        data.seek(0)
        test = np.genfromtxt(data, usecols=(1,), converters={1: lambda s: int(s, 16)})
        assert_equal(test, [33, 66])

    def test_invalid_converter(self):
        strip_rand = lambda x: float(b'r' in x.lower() and x.split()[-1] or (b'r' not in x.lower() and x.strip() or 0.0))
        strip_per = lambda x: float(b'%' in x.lower() and x.split()[0] or (b'%' not in x.lower() and x.strip() or 0.0))
        s = TextIO('D01N01,10/1/2003 ,1 %,R 75,400,600\r\nL24U05,12/5/2003, 2 %,1,300, 150.5\r\nD02N03,10/10/2004,R 1,,7,145.55')
        kwargs = dict(converters={2: strip_per, 3: strip_rand}, delimiter=',', dtype=None)
        assert_raises(ConverterError, np.genfromtxt, s, **kwargs)

    def test_tricky_converter_bug1666(self):
        s = TextIO('q1,2\nq3,4')
        cnv = lambda s: float(s[1:])
        test = np.genfromtxt(s, delimiter=',', converters={0: cnv})
        control = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_equal(test, control)

    def test_dtype_with_converters(self):
        dstr = '2009; 23; 46'
        test = np.genfromtxt(TextIO(dstr), delimiter=';', dtype=float, converters={0: bytes})
        control = np.array([('2009', 23.0, 46)], dtype=[('f0', '|S4'), ('f1', float), ('f2', float)])
        assert_equal(test, control)
        test = np.genfromtxt(TextIO(dstr), delimiter=';', dtype=float, converters={0: float})
        control = np.array([2009.0, 23.0, 46])
        assert_equal(test, control)

    def test_dtype_with_converters_and_usecols(self):
        dstr = '1,5,-1,1:1\n2,8,-1,1:n\n3,3,-2,m:n\n'
        dmap = {'1:1': 0, '1:n': 1, 'm:1': 2, 'm:n': 3}
        dtyp = [('e1', 'i4'), ('e2', 'i4'), ('e3', 'i2'), ('n', 'i1')]
        conv = {0: int, 1: int, 2: int, 3: lambda r: dmap[r.decode()]}
        test = np.recfromcsv(TextIO(dstr), dtype=dtyp, delimiter=',', names=None, converters=conv)
        control = np.rec.array([(1, 5, -1, 0), (2, 8, -1, 1), (3, 3, -2, 3)], dtype=dtyp)
        assert_equal(test, control)
        dtyp = [('e1', 'i4'), ('e2', 'i4'), ('n', 'i1')]
        test = np.recfromcsv(TextIO(dstr), dtype=dtyp, delimiter=',', usecols=(0, 1, 3), names=None, converters=conv)
        control = np.rec.array([(1, 5, 0), (2, 8, 1), (3, 3, 3)], dtype=dtyp)
        assert_equal(test, control)

    def test_dtype_with_object(self):
        data = ' 1; 2001-01-01\n                   2; 2002-01-31 '
        ndtype = [('idx', int), ('code', object)]
        func = lambda s: strptime(s.strip(), '%Y-%m-%d')
        converters = {1: func}
        test = np.genfromtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)
        control = np.array([(1, datetime(2001, 1, 1)), (2, datetime(2002, 1, 31))], dtype=ndtype)
        assert_equal(test, control)
        ndtype = [('nest', [('idx', int), ('code', object)])]
        with assert_raises_regex(NotImplementedError, 'Nested fields.* not supported.*'):
            test = np.genfromtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)
        ndtype = [('idx', int), ('code', object), ('nest', [])]
        with assert_raises_regex(NotImplementedError, 'Nested fields.* not supported.*'):
            test = np.genfromtxt(TextIO(data), delimiter=';', dtype=ndtype, converters=converters)

    def test_dtype_with_object_no_converter(self):
        parsed = np.genfromtxt(TextIO('1'), dtype=object)
        assert parsed[()] == b'1'
        parsed = np.genfromtxt(TextIO('string'), dtype=object)
        assert parsed[()] == b'string'

    def test_userconverters_with_explicit_dtype(self):
        data = TextIO('skip,skip,2001-01-01,1.0,skip')
        test = np.genfromtxt(data, delimiter=',', names=None, dtype=float, usecols=(2, 3), converters={2: bytes})
        control = np.array([('2001-01-01', 1.0)], dtype=[('', '|S10'), ('', float)])
        assert_equal(test, control)

    def test_utf8_userconverters_with_explicit_dtype(self):
        utf8 = b'\xcf\x96'
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write(b'skip,skip,2001-01-01' + utf8 + b',1.0,skip')
            test = np.genfromtxt(path, delimiter=',', names=None, dtype=float, usecols=(2, 3), converters={2: np.compat.unicode}, encoding='UTF-8')
        control = np.array([('2001-01-01' + utf8.decode('UTF-8'), 1.0)], dtype=[('', '|U11'), ('', float)])
        assert_equal(test, control)

    def test_spacedelimiter(self):
        data = TextIO('1  2  3  4   5\n6  7  8  9  10')
        test = np.genfromtxt(data)
        control = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        assert_equal(test, control)

    def test_integer_delimiter(self):
        data = '  1  2  3\n  4  5 67\n890123  4'
        test = np.genfromtxt(TextIO(data), delimiter=3)
        control = np.array([[1, 2, 3], [4, 5, 67], [890, 123, 4]])
        assert_equal(test, control)

    def test_missing(self):
        data = TextIO('1,2,3,,5\n')
        test = np.genfromtxt(data, dtype=int, delimiter=',', converters={3: lambda s: int(s or -999)})
        control = np.array([1, 2, 3, -999, 5], int)
        assert_equal(test, control)

    def test_missing_with_tabs(self):
        txt = '1\t2\t3\n\t2\t\n1\t\t3'
        test = np.genfromtxt(TextIO(txt), delimiter='\t', usemask=True)
        ctrl_d = np.array([(1, 2, 3), (np.nan, 2, np.nan), (1, np.nan, 3)])
        ctrl_m = np.array([(0, 0, 0), (1, 0, 1), (0, 1, 0)], dtype=bool)
        assert_equal(test.data, ctrl_d)
        assert_equal(test.mask, ctrl_m)

    def test_usecols(self):
        control = np.array([[1, 2], [3, 4]], float)
        data = TextIO()
        np.savetxt(data, control)
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=(1,))
        assert_equal(test, control[:, 1])
        control = np.array([[1, 2, 3], [3, 4, 5]], float)
        data = TextIO()
        np.savetxt(data, control)
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=(1, 2))
        assert_equal(test, control[:, 1:])
        data.seek(0)
        test = np.genfromtxt(data, dtype=float, usecols=np.array([1, 2]))
        assert_equal(test, control[:, 1:])

    def test_usecols_as_css(self):
        data = '1 2 3\n4 5 6'
        test = np.genfromtxt(TextIO(data), names='a, b, c', usecols='a, c')
        ctrl = np.array([(1, 3), (4, 6)], dtype=[(_, float) for _ in 'ac'])
        assert_equal(test, ctrl)

    def test_usecols_with_structured_dtype(self):
        data = TextIO('JOE 70.1 25.3\nBOB 60.5 27.9')
        names = ['stid', 'temp']
        dtypes = ['S4', 'f8']
        test = np.genfromtxt(data, usecols=(0, 2), dtype=list(zip(names, dtypes)))
        assert_equal(test['stid'], [b'JOE', b'BOB'])
        assert_equal(test['temp'], [25.3, 27.9])

    def test_usecols_with_integer(self):
        test = np.genfromtxt(TextIO(b'1 2 3\n4 5 6'), usecols=0)
        assert_equal(test, np.array([1.0, 4.0]))

    def test_usecols_with_named_columns(self):
        ctrl = np.array([(1, 3), (4, 6)], dtype=[('a', float), ('c', float)])
        data = '1 2 3\n4 5 6'
        kwargs = dict(names='a, b, c')
        test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(data), usecols=('a', 'c'), **kwargs)
        assert_equal(test, ctrl)

    def test_empty_file(self):
        with suppress_warnings() as sup:
            sup.filter(message='genfromtxt: Empty input file:')
            data = TextIO()
            test = np.genfromtxt(data)
            assert_equal(test, np.array([]))
            test = np.genfromtxt(data, skip_header=1)
            assert_equal(test, np.array([]))

    def test_fancy_dtype_alt(self):
        data = TextIO('1,2,3.0\n4,5,6.0\n')
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = np.genfromtxt(data, dtype=fancydtype, delimiter=',', usemask=True)
        control = ma.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=fancydtype)
        assert_equal(test, control)

    def test_shaped_dtype(self):
        c = TextIO('aaaa  1.0  8.0  1 2 3 4 5 6')
        dt = np.dtype([('name', 'S4'), ('x', float), ('y', float), ('block', int, (2, 3))])
        x = np.genfromtxt(c, dtype=dt)
        a = np.array([('aaaa', 1.0, 8.0, [[1, 2, 3], [4, 5, 6]])], dtype=dt)
        assert_array_equal(x, a)

    def test_withmissing(self):
        data = TextIO('A,B\n0,1\n2,N/A')
        kwargs = dict(delimiter=',', missing_values='N/A', names=True)
        test = np.genfromtxt(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)], mask=[(False, False), (False, True)], dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        data.seek(0)
        test = np.genfromtxt(data, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)], mask=[(False, False), (False, True)], dtype=[('A', float), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_user_missing_values(self):
        data = 'A, B, C\n0, 0., 0j\n1, N/A, 1j\n-9, 2.2, N/A\n3, -99, 3j'
        basekwargs = dict(dtype=None, delimiter=',', names=True)
        mdtype = [('A', int), ('B', float), ('C', complex)]
        test = np.genfromtxt(TextIO(data), missing_values='N/A', **basekwargs)
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j), (-9, 2.2, -999j), (3, -99, 3j)], mask=[(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)], dtype=mdtype)
        assert_equal(test, control)
        basekwargs['dtype'] = mdtype
        test = np.genfromtxt(TextIO(data), missing_values={0: -9, 1: -99, 2: -999j}, usemask=True, **basekwargs)
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j), (-9, 2.2, -999j), (3, -99, 3j)], mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)], dtype=mdtype)
        assert_equal(test, control)
        test = np.genfromtxt(TextIO(data), missing_values={0: -9, 'B': -99, 'C': -999j}, usemask=True, **basekwargs)
        control = ma.array([(0, 0.0, 0j), (1, -999, 1j), (-9, 2.2, -999j), (3, -99, 3j)], mask=[(0, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 0)], dtype=mdtype)
        assert_equal(test, control)

    def test_user_filling_values(self):
        ctrl = np.array([(0, 3), (4, -999)], dtype=[('a', int), ('b', int)])
        data = 'N/A, 2, 3\n4, ,???'
        kwargs = dict(delimiter=',', dtype=int, names='a,b,c', missing_values={0: 'N/A', 'b': ' ', 2: '???'}, filling_values={0: 0, 'b': 0, 2: -999})
        test = np.genfromtxt(TextIO(data), **kwargs)
        ctrl = np.array([(0, 2, 3), (4, 0, -999)], dtype=[(_, int) for _ in 'abc'])
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(data), usecols=(0, -1), **kwargs)
        ctrl = np.array([(0, 3), (4, -999)], dtype=[(_, int) for _ in 'ac'])
        assert_equal(test, ctrl)
        data2 = '1,2,*,4\n5,*,7,8\n'
        test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int, missing_values='*', filling_values=0)
        ctrl = np.array([[1, 2, 0, 4], [5, 0, 7, 8]])
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(data2), delimiter=',', dtype=int, missing_values='*', filling_values=-1)
        ctrl = np.array([[1, 2, -1, 4], [5, -1, 7, 8]])
        assert_equal(test, ctrl)

    def test_withmissing_float(self):
        data = TextIO('A,B\n0,1.5\n2,-999.00')
        test = np.genfromtxt(data, dtype=None, delimiter=',', missing_values='-999.0', names=True, usemask=True)
        control = ma.array([(0, 1.5), (2, -1.0)], mask=[(False, False), (False, True)], dtype=[('A', int), ('B', float)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)

    def test_with_masked_column_uniform(self):
        data = TextIO('1 2 3\n4 5 6\n')
        test = np.genfromtxt(data, dtype=None, missing_values='2,5', usemask=True)
        control = ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [0, 1, 0]])
        assert_equal(test, control)

    def test_with_masked_column_various(self):
        data = TextIO('True 2 3\nFalse 5 6\n')
        test = np.genfromtxt(data, dtype=None, missing_values='2,5', usemask=True)
        control = ma.array([(1, 2, 3), (0, 5, 6)], mask=[(0, 1, 0), (0, 1, 0)], dtype=[('f0', bool), ('f1', bool), ('f2', int)])
        assert_equal(test, control)

    def test_invalid_raise(self):
        data = ['1, 1, 1, 1, 1'] * 50
        for i in range(5):
            data[10 * i] = '2, 2, 2, 2 2'
        data.insert(0, 'a, b, c, d, e')
        mdata = TextIO('\n'.join(data))
        kwargs = dict(delimiter=',', dtype=None, names=True)

        def f():
            return np.genfromtxt(mdata, invalid_raise=False, **kwargs)
        mtest = assert_warns(ConversionWarning, f)
        assert_equal(len(mtest), 45)
        assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'abcde']))
        mdata.seek(0)
        assert_raises(ValueError, np.genfromtxt, mdata, delimiter=',', names=True)

    def test_invalid_raise_with_usecols(self):
        data = ['1, 1, 1, 1, 1'] * 50
        for i in range(5):
            data[10 * i] = '2, 2, 2, 2 2'
        data.insert(0, 'a, b, c, d, e')
        mdata = TextIO('\n'.join(data))
        kwargs = dict(delimiter=',', dtype=None, names=True, invalid_raise=False)

        def f():
            return np.genfromtxt(mdata, usecols=(0, 4), **kwargs)
        mtest = assert_warns(ConversionWarning, f)
        assert_equal(len(mtest), 45)
        assert_equal(mtest, np.ones(45, dtype=[(_, int) for _ in 'ae']))
        mdata.seek(0)
        mtest = np.genfromtxt(mdata, usecols=(0, 1), **kwargs)
        assert_equal(len(mtest), 50)
        control = np.ones(50, dtype=[(_, int) for _ in 'ab'])
        control[[10 * _ for _ in range(5)]] = (2, 2)
        assert_equal(mtest, control)

    def test_inconsistent_dtype(self):
        data = ['1, 1, 1, 1, -1.1'] * 50
        mdata = TextIO('\n'.join(data))
        converters = {4: lambda x: '(%s)' % x.decode()}
        kwargs = dict(delimiter=',', converters=converters, dtype=[(_, int) for _ in 'abcde'])
        assert_raises(ValueError, np.genfromtxt, mdata, **kwargs)

    def test_default_field_format(self):
        data = '0, 1, 2.3\n4, 5, 6.7'
        mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=None, defaultfmt='f%02i')
        ctrl = np.array([(0, 1, 2.3), (4, 5, 6.7)], dtype=[('f00', int), ('f01', int), ('f02', float)])
        assert_equal(mtest, ctrl)

    def test_single_dtype_wo_names(self):
        data = '0, 1, 2.3\n4, 5, 6.7'
        mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=float, defaultfmt='f%02i')
        ctrl = np.array([[0.0, 1.0, 2.3], [4.0, 5.0, 6.7]], dtype=float)
        assert_equal(mtest, ctrl)

    def test_single_dtype_w_explicit_names(self):
        data = '0, 1, 2.3\n4, 5, 6.7'
        mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=float, names='a, b, c')
        ctrl = np.array([(0.0, 1.0, 2.3), (4.0, 5.0, 6.7)], dtype=[(_, float) for _ in 'abc'])
        assert_equal(mtest, ctrl)

    def test_single_dtype_w_implicit_names(self):
        data = 'a, b, c\n0, 1, 2.3\n4, 5, 6.7'
        mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=float, names=True)
        ctrl = np.array([(0.0, 1.0, 2.3), (4.0, 5.0, 6.7)], dtype=[(_, float) for _ in 'abc'])
        assert_equal(mtest, ctrl)

    def test_easy_structured_dtype(self):
        data = '0, 1, 2.3\n4, 5, 6.7'
        mtest = np.genfromtxt(TextIO(data), delimiter=',', dtype=(int, float, float), defaultfmt='f_%02i')
        ctrl = np.array([(0, 1.0, 2.3), (4, 5.0, 6.7)], dtype=[('f_00', int), ('f_01', float), ('f_02', float)])
        assert_equal(mtest, ctrl)

    def test_autostrip(self):
        data = '01/01/2003  , 1.3,   abcde'
        kwargs = dict(delimiter=',', dtype=None)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            mtest = np.genfromtxt(TextIO(data), **kwargs)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        ctrl = np.array([('01/01/2003  ', 1.3, '   abcde')], dtype=[('f0', '|S12'), ('f1', float), ('f2', '|S8')])
        assert_equal(mtest, ctrl)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            mtest = np.genfromtxt(TextIO(data), autostrip=True, **kwargs)
            assert_(w[0].category is np.VisibleDeprecationWarning)
        ctrl = np.array([('01/01/2003', 1.3, 'abcde')], dtype=[('f0', '|S10'), ('f1', float), ('f2', '|S5')])
        assert_equal(mtest, ctrl)

    def test_replace_space(self):
        txt = 'A.A, B (B), C:C\n1, 2, 3.14'
        test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=None)
        ctrl_dtype = [('AA', int), ('B_B', int), ('CC', float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=None, replace_space='', deletechars='')
        ctrl_dtype = [('A.A', int), ('B (B)', int), ('C:C', float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=None, deletechars='')
        ctrl_dtype = [('A.A', int), ('B_(B)', int), ('C:C', float)]
        ctrl = np.array((1, 2, 3.14), dtype=ctrl_dtype)
        assert_equal(test, ctrl)

    def test_replace_space_known_dtype(self):
        txt = 'A.A, B (B), C:C\n1, 2, 3'
        test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=int)
        ctrl_dtype = [('AA', int), ('B_B', int), ('CC', int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=int, replace_space='', deletechars='')
        ctrl_dtype = [('A.A', int), ('B (B)', int), ('C:C', int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)
        test = np.genfromtxt(TextIO(txt), delimiter=',', names=True, dtype=int, deletechars='')
        ctrl_dtype = [('A.A', int), ('B_(B)', int), ('C:C', int)]
        ctrl = np.array((1, 2, 3), dtype=ctrl_dtype)
        assert_equal(test, ctrl)

    def test_incomplete_names(self):
        data = 'A,,C\n0,1,2\n3,4,5'
        kwargs = dict(delimiter=',', names=True)
        ctrl = np.array([(0, 1, 2), (3, 4, 5)], dtype=[(_, int) for _ in ('A', 'f0', 'C')])
        test = np.genfromtxt(TextIO(data), dtype=None, **kwargs)
        assert_equal(test, ctrl)
        ctrl = np.array([(0, 1, 2), (3, 4, 5)], dtype=[(_, float) for _ in ('A', 'f0', 'C')])
        test = np.genfromtxt(TextIO(data), **kwargs)

    def test_names_auto_completion(self):
        data = '1 2 3\n 4 5 6'
        test = np.genfromtxt(TextIO(data), dtype=(int, float, int), names='a')
        ctrl = np.array([(1, 2, 3), (4, 5, 6)], dtype=[('a', int), ('f0', float), ('f1', int)])
        assert_equal(test, ctrl)

    def test_names_with_usecols_bug1636(self):
        data = 'A,B,C,D,E\n0,1,2,3,4\n0,1,2,3,4\n0,1,2,3,4'
        ctrl_names = ('A', 'C', 'E')
        test = np.genfromtxt(TextIO(data), dtype=(int, int, int), delimiter=',', usecols=(0, 2, 4), names=True)
        assert_equal(test.dtype.names, ctrl_names)
        test = np.genfromtxt(TextIO(data), dtype=(int, int, int), delimiter=',', usecols=('A', 'C', 'E'), names=True)
        assert_equal(test.dtype.names, ctrl_names)
        test = np.genfromtxt(TextIO(data), dtype=int, delimiter=',', usecols=('A', 'C', 'E'), names=True)
        assert_equal(test.dtype.names, ctrl_names)

    def test_fixed_width_names(self):
        data = '    A    B   C\n    0    1 2.3\n   45   67   9.'
        kwargs = dict(delimiter=(5, 5, 4), names=True, dtype=None)
        ctrl = np.array([(0, 1, 2.3), (45, 67, 9.0)], dtype=[('A', int), ('B', int), ('C', float)])
        test = np.genfromtxt(TextIO(data), **kwargs)
        assert_equal(test, ctrl)
        kwargs = dict(delimiter=5, names=True, dtype=None)
        ctrl = np.array([(0, 1, 2.3), (45, 67, 9.0)], dtype=[('A', int), ('B', int), ('C', float)])
        test = np.genfromtxt(TextIO(data), **kwargs)
        assert_equal(test, ctrl)

    def test_filling_values(self):
        data = b'1, 2, 3\n1, , 5\n0, 6, \n'
        kwargs = dict(delimiter=',', dtype=None, filling_values=-999)
        ctrl = np.array([[1, 2, 3], [1, -999, 5], [0, 6, -999]], dtype=int)
        test = np.genfromtxt(TextIO(data), **kwargs)
        assert_equal(test, ctrl)

    def test_comments_is_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO('test1,testNonetherestofthedata'), dtype=None, comments=None, delimiter=',')
            assert_(w[0].category is np.VisibleDeprecationWarning)
        assert_equal(test[1], b'testNonetherestofthedata')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO('test1, testNonetherestofthedata'), dtype=None, comments=None, delimiter=',')
            assert_(w[0].category is np.VisibleDeprecationWarning)
        assert_equal(test[1], b' testNonetherestofthedata')

    def test_latin1(self):
        latin1 = b'\xf6\xfc\xf6'
        norm = b'norm1,norm2,norm3\n'
        enc = b'test1,testNonethe' + latin1 + b',test3\n'
        s = norm + enc + norm
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO(s), dtype=None, comments=None, delimiter=',')
            assert_(w[0].category is np.VisibleDeprecationWarning)
        assert_equal(test[1, 0], b'test1')
        assert_equal(test[1, 1], b'testNonethe' + latin1)
        assert_equal(test[1, 2], b'test3')
        test = np.genfromtxt(TextIO(s), dtype=None, comments=None, delimiter=',', encoding='latin1')
        assert_equal(test[1, 0], 'test1')
        assert_equal(test[1, 1], 'testNonethe' + latin1.decode('latin1'))
        assert_equal(test[1, 2], 'test3')
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO(b'0,testNonethe' + latin1), dtype=None, comments=None, delimiter=',')
            assert_(w[0].category is np.VisibleDeprecationWarning)
        assert_equal(test['f0'], 0)
        assert_equal(test['f1'], b'testNonethe' + latin1)

    def test_binary_decode_autodtype(self):
        utf16 = b'\xff\xfeh\x04 \x00i\x04 \x00j\x04'
        v = self.loadfunc(BytesIO(utf16), dtype=None, encoding='UTF-16')
        assert_array_equal(v, np.array(utf16.decode('UTF-16').split()))

    def test_utf8_byte_encoding(self):
        utf8 = b'\xcf\x96'
        norm = b'norm1,norm2,norm3\n'
        enc = b'test1,testNonethe' + utf8 + b',test3\n'
        s = norm + enc + norm
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
            test = np.genfromtxt(TextIO(s), dtype=None, comments=None, delimiter=',')
            assert_(w[0].category is np.VisibleDeprecationWarning)
        ctl = np.array([[b'norm1', b'norm2', b'norm3'], [b'test1', b'testNonethe' + utf8, b'test3'], [b'norm1', b'norm2', b'norm3']])
        assert_array_equal(test, ctl)

    def test_utf8_file(self):
        utf8 = b'\xcf\x96'
        with temppath() as path:
            with open(path, 'wb') as f:
                f.write((b'test1,testNonethe' + utf8 + b',test3\n') * 2)
            test = np.genfromtxt(path, dtype=None, comments=None, delimiter=',', encoding='UTF-8')
            ctl = np.array([['test1', 'testNonethe' + utf8.decode('UTF-8'), 'test3'], ['test1', 'testNonethe' + utf8.decode('UTF-8'), 'test3']], dtype=np.str_)
            assert_array_equal(test, ctl)
            with open(path, 'wb') as f:
                f.write(b'0,testNonethe' + utf8)
            test = np.genfromtxt(path, dtype=None, comments=None, delimiter=',', encoding='UTF-8')
            assert_equal(test['f0'], 0)
            assert_equal(test['f1'], 'testNonethe' + utf8.decode('UTF-8'))

    def test_utf8_file_nodtype_unicode(self):
        utf8 = 'ϖ'
        latin1 = 'öüö'
        try:
            encoding = locale.getpreferredencoding()
            utf8.encode(encoding)
        except (UnicodeError, ImportError):
            pytest.skip('Skipping test_utf8_file_nodtype_unicode, unable to encode utf8 in preferred encoding')
        with temppath() as path:
            with io.open(path, 'wt') as f:
                f.write('norm1,norm2,norm3\n')
                f.write('norm1,' + latin1 + ',norm3\n')
                f.write('test1,testNonethe' + utf8 + ',test3\n')
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('always', '', np.VisibleDeprecationWarning)
                test = np.genfromtxt(path, dtype=None, comments=None, delimiter=',')
                assert_(w[0].category is np.VisibleDeprecationWarning)
            ctl = np.array([['norm1', 'norm2', 'norm3'], ['norm1', latin1, 'norm3'], ['test1', 'testNonethe' + utf8, 'test3']], dtype=np.str_)
            assert_array_equal(test, ctl)

    def test_recfromtxt(self):
        data = TextIO('A,B\n0,1\n2,3')
        kwargs = dict(delimiter=',', missing_values='N/A', names=True)
        test = np.recfromtxt(data, **kwargs)
        control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        data = TextIO('A,B\n0,1\n2,N/A')
        test = np.recfromtxt(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)], mask=[(False, False), (False, True)], dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.A, [0, 2])

    def test_recfromcsv(self):
        data = TextIO('A,B\n0,1\n2,3')
        kwargs = dict(missing_values='N/A', names=True, case_sensitive=True)
        test = np.recfromcsv(data, dtype=None, **kwargs)
        control = np.array([(0, 1), (2, 3)], dtype=[('A', int), ('B', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        data = TextIO('A,B\n0,1\n2,N/A')
        test = np.recfromcsv(data, dtype=None, usemask=True, **kwargs)
        control = ma.array([(0, 1), (2, -1)], mask=[(False, False), (False, True)], dtype=[('A', int), ('B', int)])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(test.A, [0, 2])
        data = TextIO('A,B\n0,1\n2,3')
        test = np.recfromcsv(data, missing_values='N/A')
        control = np.array([(0, 1), (2, 3)], dtype=[('a', int), ('b', int)])
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        data = TextIO('A,B\n0,1\n2,3')
        dtype = [('a', int), ('b', float)]
        test = np.recfromcsv(data, missing_values='N/A', dtype=dtype)
        control = np.array([(0, 1), (2, 3)], dtype=dtype)
        assert_(isinstance(test, np.recarray))
        assert_equal(test, control)
        data = TextIO('color\n"red"\n"blue"')
        test = np.recfromcsv(data, converters={0: lambda x: x.strip(b'"')})
        control = np.array([('red',), ('blue',)], dtype=[('color', (bytes, 4))])
        assert_equal(test.dtype, control.dtype)
        assert_equal(test, control)

    def test_max_rows(self):
        data = '1 2\n3 4\n5 6\n7 8\n9 10\n'
        txt = TextIO(data)
        a1 = np.genfromtxt(txt, max_rows=3)
        a2 = np.genfromtxt(txt)
        assert_equal(a1, [[1, 2], [3, 4], [5, 6]])
        assert_equal(a2, [[7, 8], [9, 10]])
        assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=0)
        data = '1 1\n2 2\n0 \n3 3\n4 4\n5  \n6  \n7  \n'
        test = np.genfromtxt(TextIO(data), max_rows=2)
        control = np.array([[1.0, 1.0], [2.0, 2.0]])
        assert_equal(test, control)
        assert_raises(ValueError, np.genfromtxt, TextIO(data), skip_footer=1, max_rows=4)
        assert_raises(ValueError, np.genfromtxt, TextIO(data), max_rows=4)
        with suppress_warnings() as sup:
            sup.filter(ConversionWarning)
            test = np.genfromtxt(TextIO(data), max_rows=4, invalid_raise=False)
            control = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
            assert_equal(test, control)
            test = np.genfromtxt(TextIO(data), max_rows=5, invalid_raise=False)
            control = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
            assert_equal(test, control)
        data = 'a b\n#c d\n1 1\n2 2\n#0 \n3 3\n4 4\n5  5\n'
        txt = TextIO(data)
        test = np.genfromtxt(txt, skip_header=1, max_rows=3, names=True)
        control = np.array([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], dtype=[('c', '<f8'), ('d', '<f8')])
        assert_equal(test, control)
        test = np.genfromtxt(txt, max_rows=None, dtype=test.dtype)
        control = np.array([(4.0, 4.0), (5.0, 5.0)], dtype=[('c', '<f8'), ('d', '<f8')])
        assert_equal(test, control)

    def test_gft_using_filename(self):
        tgt = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')
        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            with temppath() as name:
                with open(name, 'w') as f:
                    f.write(data)
                res = np.genfromtxt(name)
            assert_array_equal(res, tgt)

    def test_gft_from_gzip(self):
        wanted = np.arange(6).reshape((2, 3))
        linesep = ('\n', '\r\n', '\r')
        for sep in linesep:
            data = '0 1 2' + sep + '3 4 5'
            s = BytesIO()
            with gzip.GzipFile(fileobj=s, mode='w') as g:
                g.write(asbytes(data))
            with temppath(suffix='.gz2') as name:
                with open(name, 'w') as f:
                    f.write(data)
                assert_array_equal(np.genfromtxt(name), wanted)

    def test_gft_using_generator(self):

        def count():
            for i in range(10):
                yield asbytes('%d' % i)
        res = np.genfromtxt(count())
        assert_array_equal(res, np.arange(10))

    def test_auto_dtype_largeint(self):
        data = TextIO('73786976294838206464 17179869184 1024')
        test = np.genfromtxt(data, dtype=None)
        assert_equal(test.dtype.names, ['f0', 'f1', 'f2'])
        assert_(test.dtype['f0'] == float)
        assert_(test.dtype['f1'] == np.int64)
        assert_(test.dtype['f2'] == np.int_)
        assert_allclose(test['f0'], 7.378697629483821e+19)
        assert_equal(test['f1'], 17179869184)
        assert_equal(test['f2'], 1024)

    def test_unpack_float_data(self):
        txt = TextIO('1,2,3\n4,5,6\n7,8,9\n0.0,1.0,2.0')
        a, b, c = np.loadtxt(txt, delimiter=',', unpack=True)
        assert_array_equal(a, np.array([1.0, 4.0, 7.0, 0.0]))
        assert_array_equal(b, np.array([2.0, 5.0, 8.0, 1.0]))
        assert_array_equal(c, np.array([3.0, 6.0, 9.0, 2.0]))

    def test_unpack_structured(self):
        txt = TextIO('M 21 72\nF 35 58')
        dt = {'names': ('a', 'b', 'c'), 'formats': ('S1', 'i4', 'f4')}
        a, b, c = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_equal(a.dtype, np.dtype('S1'))
        assert_equal(b.dtype, np.dtype('i4'))
        assert_equal(c.dtype, np.dtype('f4'))
        assert_array_equal(a, np.array([b'M', b'F']))
        assert_array_equal(b, np.array([21, 35]))
        assert_array_equal(c, np.array([72.0, 58.0]))

    def test_unpack_auto_dtype(self):
        txt = TextIO('M 21 72.\nF 35 58.')
        expected = (np.array(['M', 'F']), np.array([21, 35]), np.array([72.0, 58.0]))
        test = np.genfromtxt(txt, dtype=None, unpack=True, encoding='utf-8')
        for arr, result in zip(expected, test):
            assert_array_equal(arr, result)
            assert_equal(arr.dtype, result.dtype)

    def test_unpack_single_name(self):
        txt = TextIO('21\n35')
        dt = {'names': ('a',), 'formats': ('i4',)}
        expected = np.array([21, 35], dtype=np.int32)
        test = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_array_equal(expected, test)
        assert_equal(expected.dtype, test.dtype)

    def test_squeeze_scalar(self):
        txt = TextIO('1')
        dt = {'names': ('a',), 'formats': ('i4',)}
        expected = np.array((1,), dtype=np.int32)
        test = np.genfromtxt(txt, dtype=dt, unpack=True)
        assert_array_equal(expected, test)
        assert_equal((), test.shape)
        assert_equal(expected.dtype, test.dtype)

    @pytest.mark.parametrize('ndim', [0, 1, 2])
    def test_ndmin_keyword(self, ndim: int):
        txt = '42'
        a = np.loadtxt(StringIO(txt), ndmin=ndim)
        b = np.genfromtxt(StringIO(txt), ndmin=ndim)
        assert_array_equal(a, b)