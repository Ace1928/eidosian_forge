import sys
import unittest
from test.support.os_helper import EnvironmentVarGuard
from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
@unittest.skipIf(sys.platform == 'win32', "can't test on Windows")
def test_runtime_libdir_option(self):
    sys.platform = 'darwin'
    self.assertEqual(self.cc.rpath_foo(), '-L/foo')
    sys.platform = 'hp-ux'
    old_gcv = sysconfig.get_config_var

    def gcv(v):
        return 'xxx'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), ['+s', '-L/foo'])

    def gcv(v):
        return 'gcc'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), ['-Wl,+s', '-L/foo'])

    def gcv(v):
        return 'g++'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), ['-Wl,+s', '-L/foo'])
    sysconfig.get_config_var = old_gcv
    sys.platform = 'bar'

    def gcv(v):
        if v == 'CC':
            return 'gcc'
        elif v == 'GNULD':
            return 'yes'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), '-Wl,--enable-new-dtags,-R/foo')
    sys.platform = 'bar'

    def gcv(v):
        if v == 'CC':
            return 'gcc'
        elif v == 'GNULD':
            return 'no'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), '-Wl,-R/foo')
    sys.platform = 'bar'

    def gcv(v):
        if v == 'CC':
            return 'x86_64-pc-linux-gnu-gcc-4.4.2'
        elif v == 'GNULD':
            return 'yes'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), '-Wl,--enable-new-dtags,-R/foo')
    sys.platform = 'bar'

    def gcv(v):
        if v == 'CC':
            return 'cc'
        elif v == 'GNULD':
            return 'yes'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), '-R/foo')
    sys.platform = 'bar'

    def gcv(v):
        if v == 'CC':
            return 'cc'
        elif v == 'GNULD':
            return 'no'
    sysconfig.get_config_var = gcv
    self.assertEqual(self.cc.rpath_foo(), '-R/foo')