import unittest
import unittest.mock
import os.path
import sys
import test.support
from test.support import os_helper
from ctypes import *
from ctypes.util import find_library
class Test_OpenGL_libs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lib_gl = lib_glu = lib_gle = None
        if sys.platform == 'win32':
            lib_gl = find_library('OpenGL32')
            lib_glu = find_library('Glu32')
        elif sys.platform == 'darwin':
            lib_gl = lib_glu = find_library('OpenGL')
        else:
            lib_gl = find_library('GL')
            lib_glu = find_library('GLU')
            lib_gle = find_library('gle')
        if test.support.verbose:
            print('OpenGL libraries:')
            for item in (('GL', lib_gl), ('GLU', lib_glu), ('gle', lib_gle)):
                print('\t', item)
        cls.gl = cls.glu = cls.gle = None
        if lib_gl:
            try:
                cls.gl = CDLL(lib_gl, mode=RTLD_GLOBAL)
            except OSError:
                pass
        if lib_glu:
            try:
                cls.glu = CDLL(lib_glu, RTLD_GLOBAL)
            except OSError:
                pass
        if lib_gle:
            try:
                cls.gle = CDLL(lib_gle)
            except OSError:
                pass

    @classmethod
    def tearDownClass(cls):
        cls.gl = cls.glu = cls.gle = None

    def test_gl(self):
        if self.gl is None:
            self.skipTest('lib_gl not available')
        self.gl.glClearIndex

    def test_glu(self):
        if self.glu is None:
            self.skipTest('lib_glu not available')
        self.glu.gluBeginCurve

    def test_gle(self):
        if self.gle is None:
            self.skipTest('lib_gle not available')
        self.gle.gleGetJoinStyle

    def test_shell_injection(self):
        result = find_library('; echo Hello shell > ' + os_helper.TESTFN)
        self.assertFalse(os.path.lexists(os_helper.TESTFN))
        self.assertIsNone(result)