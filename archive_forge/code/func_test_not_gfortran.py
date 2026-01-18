from numpy.testing import assert_
import numpy.distutils.fcompiler
def test_not_gfortran(self):
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu95')
    for vs, _ in g77_version_strings:
        v = fc.version_match(vs)
        assert_(v is None, (vs, v))