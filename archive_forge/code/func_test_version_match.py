from numpy.testing import assert_
import numpy.distutils.fcompiler
def test_version_match(self):
    for comp, vs, version in nag_version_strings:
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler=comp)
        v = fc.version_match(vs)
        assert_(v == version)