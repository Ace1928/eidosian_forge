import sys
import unittest
from test.support.os_helper import EnvironmentVarGuard
from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
@unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for OS X')
def test_osx_cc_overrides_ldshared(self):

    def gcv(v):
        if v == 'LDSHARED':
            return 'gcc-4.2 -bundle -undefined dynamic_lookup '
        return 'gcc-4.2'
    sysconfig.get_config_var = gcv
    with EnvironmentVarGuard() as env:
        env['CC'] = 'my_cc'
        del env['LDSHARED']
        sysconfig.customize_compiler(self.cc)
    self.assertEqual(self.cc.linker_so[0], 'my_cc')