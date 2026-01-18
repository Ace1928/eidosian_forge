import os
from numpy.distutils.npy_pkg_config import read_config, parse_flags
from numpy.testing import temppath, assert_
def test_simple_lflags(self):
    d = parse_flags('-L/usr/lib -lfoo -L/usr/lib -lbar')
    assert_(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
    assert_(d['libraries'] == ['foo', 'bar'])
    d = parse_flags('-L /usr/lib -lfoo -L/usr/lib -lbar')
    assert_(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
    assert_(d['libraries'] == ['foo', 'bar'])