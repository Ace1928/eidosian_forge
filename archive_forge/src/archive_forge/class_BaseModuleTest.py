import sys
import unittest
import platform
import pygame
class BaseModuleTest(unittest.TestCase):

    def tearDown(self):
        pygame.quit()

    def test_get_sdl_byteorder(self):
        """Ensure the SDL byte order is valid"""
        byte_order = pygame.get_sdl_byteorder()
        expected_options = (pygame.LIL_ENDIAN, pygame.BIG_ENDIAN)
        self.assertIn(byte_order, expected_options)

    def test_get_sdl_version(self):
        """Ensure the SDL version is valid"""
        self.assertEqual(len(pygame.get_sdl_version()), 3)

    class ExporterBase:

        def __init__(self, shape, typechar, itemsize):
            import ctypes
            ndim = len(shape)
            self.ndim = ndim
            self.shape = tuple(shape)
            array_len = 1
            for d in shape:
                array_len *= d
            self.size = itemsize * array_len
            self.parent = ctypes.create_string_buffer(self.size)
            self.itemsize = itemsize
            strides = [itemsize] * ndim
            for i in range(ndim - 1, 0, -1):
                strides[i - 1] = strides[i] * shape[i]
            self.strides = tuple(strides)
            self.data = (ctypes.addressof(self.parent), False)
            if self.itemsize == 1:
                byteorder = '|'
            elif sys.byteorder == 'big':
                byteorder = '>'
            else:
                byteorder = '<'
            self.typestr = byteorder + typechar + str(self.itemsize)

    def assertSame(self, proxy, obj):
        self.assertEqual(proxy.length, obj.size)
        iface = proxy.__array_interface__
        self.assertEqual(iface['typestr'], obj.typestr)
        self.assertEqual(iface['shape'], obj.shape)
        self.assertEqual(iface['strides'], obj.strides)
        self.assertEqual(iface['data'], obj.data)

    def test_PgObject_GetBuffer_array_interface(self):
        from pygame.bufferproxy import BufferProxy

        class Exporter(self.ExporterBase):

            def get__array_interface__(self):
                return {'version': 3, 'typestr': self.typestr, 'shape': self.shape, 'strides': self.strides, 'data': self.data}
            __array_interface__ = property(get__array_interface__)
            __array_struct__ = property(lambda self: None)
        _shape = [2, 3, 5, 7, 11]
        for ndim in range(1, len(_shape)):
            o = Exporter(_shape[0:ndim], 'i', 2)
            v = BufferProxy(o)
            self.assertSame(v, o)
        ndim = 2
        shape = _shape[0:ndim]
        for typechar in ('i', 'u'):
            for itemsize in (1, 2, 4, 8):
                o = Exporter(shape, typechar, itemsize)
                v = BufferProxy(o)
                self.assertSame(v, o)
        for itemsize in (4, 8):
            o = Exporter(shape, 'f', itemsize)
            v = BufferProxy(o)
            self.assertSame(v, o)
        import weakref, gc

        class NoDictError(RuntimeError):
            pass

        class WRDict(dict):
            """Weak referenceable dict"""
            pass

        class Exporter2(Exporter):

            def get__array_interface__2(self):
                self.d = WRDict(Exporter.get__array_interface__(self))
                self.dict_ref = weakref.ref(self.d)
                return self.d
            __array_interface__ = property(get__array_interface__2)

            def free_dict(self):
                self.d = None

            def is_dict_alive(self):
                try:
                    return self.dict_ref() is not None
                except AttributeError:
                    raise NoDictError('__array_interface__ is unread')
        o = Exporter2((2, 4), 'u', 4)
        v = BufferProxy(o)
        self.assertRaises(NoDictError, o.is_dict_alive)
        length = v.length
        self.assertTrue(o.is_dict_alive())
        o.free_dict()
        gc.collect()
        self.assertFalse(o.is_dict_alive())

    def test_GetView_array_struct(self):
        from pygame.bufferproxy import BufferProxy

        class Exporter(self.ExporterBase):

            def __init__(self, shape, typechar, itemsize):
                super().__init__(shape, typechar, itemsize)
                self.view = BufferProxy(self.__dict__)

            def get__array_struct__(self):
                return self.view.__array_struct__
            __array_struct__ = property(get__array_struct__)
            __array_interface__ = property(lambda self: None)
        _shape = [2, 3, 5, 7, 11]
        for ndim in range(1, len(_shape)):
            o = Exporter(_shape[0:ndim], 'i', 2)
            v = BufferProxy(o)
            self.assertSame(v, o)
        ndim = 2
        shape = _shape[0:ndim]
        for typechar in ('i', 'u'):
            for itemsize in (1, 2, 4, 8):
                o = Exporter(shape, typechar, itemsize)
                v = BufferProxy(o)
                self.assertSame(v, o)
        for itemsize in (4, 8):
            o = Exporter(shape, 'f', itemsize)
            v = BufferProxy(o)
            self.assertSame(v, o)
        try:
            from sys import getrefcount
        except ImportError:
            pass
        else:
            o = Exporter(shape, typechar, itemsize)
            self.assertEqual(getrefcount(o.__array_struct__), 1)
    if pygame.HAVE_NEWBUF:
        from pygame.tests.test_utils import buftools

    def NEWBUF_assertSame(self, proxy, exp):
        buftools = self.buftools
        Importer = buftools.Importer
        self.assertEqual(proxy.length, exp.len)
        imp = Importer(proxy, buftools.PyBUF_RECORDS_RO)
        self.assertEqual(imp.readonly, exp.readonly)
        self.assertEqual(imp.format, exp.format)
        self.assertEqual(imp.itemsize, exp.itemsize)
        self.assertEqual(imp.ndim, exp.ndim)
        self.assertEqual(imp.shape, exp.shape)
        self.assertEqual(imp.strides, exp.strides)
        self.assertTrue(imp.suboffsets is None)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'pypy no likey')
    def test_newbuf(self):
        from pygame.bufferproxy import BufferProxy
        Exporter = self.buftools.Exporter
        _shape = [2, 3, 5, 7, 11]
        for ndim in range(1, len(_shape)):
            o = Exporter(_shape[0:ndim], '=h')
            v = BufferProxy(o)
            self.NEWBUF_assertSame(v, o)
        ndim = 2
        shape = _shape[0:ndim]
        for format in ['b', 'B', '=h', '=H', '=i', '=I', '=q', '=Q', 'f', 'd', '1h', '=1h', 'x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x']:
            o = Exporter(shape, format)
            v = BufferProxy(o)
            self.NEWBUF_assertSame(v, o)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    def test_bad_format(self):
        from pygame.bufferproxy import BufferProxy
        from pygame.newbuffer import BufferMixin
        from ctypes import create_string_buffer, addressof
        buftools = self.buftools
        Exporter = buftools.Exporter
        Importer = buftools.Importer
        PyBUF_FORMAT = buftools.PyBUF_FORMAT
        for format in ['', '=', '1', ' ', '2h', '=2h', '0x', '11x', '=!', 'h ', ' h', 'hh', '?']:
            exp = Exporter((1,), format, itemsize=2)
            b = BufferProxy(exp)
            self.assertRaises(ValueError, Importer, b, PyBUF_FORMAT)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'fails on pypy')
    def test_PgDict_AsBuffer_PyBUF_flags(self):
        from pygame.bufferproxy import BufferProxy
        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        fsys, frev = ('<', '>') if is_lil_endian else ('>', '<')
        buftools = self.buftools
        Importer = buftools.Importer
        a = BufferProxy({'typestr': '|u4', 'shape': (10, 2), 'data': (9, False)})
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 4)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 9)
        b = Importer(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 4)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 9)
        b = Importer(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, 4)
        self.assertEqual(b.shape, (10, 2))
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 9)
        a = BufferProxy({'typestr': fsys + 'i2', 'shape': (5, 10), 'strides': (24, 2), 'data': (42, False)})
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, 2)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, 100)
        self.assertEqual(b.itemsize, 2)
        self.assertEqual(b.shape, (5, 10))
        self.assertEqual(b.strides, (24, 2))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 42)
        b = Importer(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, 2)
        self.assertEqual(b.format, '=h')
        self.assertEqual(b.len, 100)
        self.assertEqual(b.itemsize, 2)
        self.assertEqual(b.shape, (5, 10))
        self.assertEqual(b.strides, (24, 2))
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, 42)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_CONTIG)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_CONTIG)
        a = BufferProxy({'typestr': frev + 'i2', 'shape': (3, 5, 10), 'strides': (120, 24, 2), 'data': (1000000, True)})
        b = Importer(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.format, frev + 'h')
        self.assertEqual(b.len, 300)
        self.assertEqual(b.itemsize, 2)
        self.assertEqual(b.shape, (3, 5, 10))
        self.assertEqual(b.strides, (120, 24, 2))
        self.assertTrue(b.suboffsets is None)
        self.assertTrue(b.readonly)
        self.assertEqual(b.buf, 1000000)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FULL)

    @unittest.skipIf(IS_PYPY or not pygame.HAVE_NEWBUF, 'newbuf with ctypes')
    def test_PgObject_AsBuffer_PyBUF_flags(self):
        from pygame.bufferproxy import BufferProxy
        import ctypes
        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        fsys, frev = ('<', '>') if is_lil_endian else ('>', '<')
        buftools = self.buftools
        Importer = buftools.Importer
        e = arrinter.Exporter((10, 2), typekind='f', itemsize=ctypes.sizeof(ctypes.c_double))
        a = BufferProxy(e)
        b = Importer(a, buftools.PyBUF_SIMPLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        b = Importer(a, buftools.PyBUF_WRITABLE)
        self.assertEqual(b.ndim, 0)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertTrue(b.shape is None)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        b = Importer(a, buftools.PyBUF_ND)
        self.assertEqual(b.ndim, e.nd)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, a.length)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertTrue(b.strides is None)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        e = arrinter.Exporter((5, 10), typekind='i', itemsize=2, strides=(24, 2))
        a = BufferProxy(e)
        b = Importer(a, buftools.PyBUF_STRIDES)
        self.assertEqual(b.ndim, e.nd)
        self.assertTrue(b.format is None)
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertEqual(b.strides, e.strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        b = Importer(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, e.nd)
        self.assertEqual(b.format, '=h')
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertEqual(b.strides, e.strides)
        self.assertTrue(b.suboffsets is None)
        self.assertFalse(b.readonly)
        self.assertEqual(b.buf, e.data)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_WRITABLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_CONTIG)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_SIMPLE)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ND)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_C_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_F_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_CONTIG)
        e = arrinter.Exporter((3, 5, 10), typekind='i', itemsize=2, strides=(120, 24, 2), flags=arrinter.PAI_ALIGNED)
        a = BufferProxy(e)
        b = Importer(a, buftools.PyBUF_FULL_RO)
        self.assertEqual(b.ndim, e.nd)
        self.assertEqual(b.format, frev + 'h')
        self.assertEqual(b.len, e.len)
        self.assertEqual(b.itemsize, e.itemsize)
        self.assertEqual(b.shape, e.shape)
        self.assertEqual(b.strides, e.strides)
        self.assertTrue(b.suboffsets is None)
        self.assertTrue(b.readonly)
        self.assertEqual(b.buf, e.data)
        self.assertRaises(BufferError, Importer, a, buftools.PyBUF_FULL)

    def test_PgObject_GetBuffer_exception(self):
        from pygame.bufferproxy import BufferProxy
        bp = BufferProxy(1)
        self.assertRaises(ValueError, getattr, bp, 'length')

    def not_init_assertions(self):
        self.assertFalse(pygame.get_init(), "pygame shouldn't be initialized")
        self.assertFalse(pygame.display.get_init(), "display shouldn't be initialized")
        if 'pygame.mixer' in sys.modules:
            self.assertFalse(pygame.mixer.get_init(), "mixer shouldn't be initialized")
        if 'pygame.font' in sys.modules:
            self.assertFalse(pygame.font.get_init(), "init shouldn't be initialized")
        import platform
        if platform.system().startswith('Darwin'):
            return
        try:
            self.assertRaises(pygame.error, pygame.scrap.get)
        except NotImplementedError:
            pass

    def init_assertions(self):
        self.assertTrue(pygame.get_init())
        self.assertTrue(pygame.display.get_init())
        if 'pygame.mixer' in sys.modules:
            self.assertTrue(pygame.mixer.get_init())
        if 'pygame.font' in sys.modules:
            self.assertTrue(pygame.font.get_init())

    def test_quit__and_init(self):
        self.not_init_assertions()
        pygame.init()
        self.init_assertions()
        pygame.quit()
        self.not_init_assertions()

    def test_register_quit(self):
        """Ensure that a registered function is called on quit()"""
        self.assertEqual(quit_count, 0)
        pygame.init()
        pygame.register_quit(quit_hook)
        pygame.quit()
        self.assertEqual(quit_count, 1)

    def test_get_error(self):
        e = pygame.get_error()
        pygame.set_error('hi')
        self.assertEqual(pygame.get_error(), 'hi')
        pygame.set_error('')
        self.assertEqual(pygame.get_error(), '')

    def test_set_error(self):
        e = pygame.get_error()
        pygame.set_error('hi')
        self.assertEqual(pygame.get_error(), 'hi')
        pygame.set_error('')
        self.assertEqual(pygame.get_error(), '')

    def test_unicode_error(self):
        pygame.set_error('你好')
        self.assertEqual('你好', pygame.get_error())

    def test_init(self):
        """Ensures init() works properly."""
        self.not_init_assertions()
        expected_min_passes = 2
        expected_fails = 0
        passes, fails = pygame.init()
        self.init_assertions()
        self.assertGreaterEqual(passes, expected_min_passes)
        self.assertEqual(fails, expected_fails)

    def test_get_init(self):
        self.assertFalse(pygame.get_init())

    def test_get_init__after_init(self):
        pygame.init()
        self.assertTrue(pygame.get_init())

    def test_get_init__after_quit(self):
        pygame.init()
        pygame.quit()
        self.assertFalse(pygame.get_init())