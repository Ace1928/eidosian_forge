import numpy as np
from numpy.testing import (
from numpy.lib.type_check import (
class TestMintypecode:

    def test_default_1(self):
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype), 'd')
        assert_equal(mintypecode('f'), 'f')
        assert_equal(mintypecode('d'), 'd')
        assert_equal(mintypecode('F'), 'F')
        assert_equal(mintypecode('D'), 'D')

    def test_default_2(self):
        for itype in '1bcsuwil':
            assert_equal(mintypecode(itype + 'f'), 'f')
            assert_equal(mintypecode(itype + 'd'), 'd')
            assert_equal(mintypecode(itype + 'F'), 'F')
            assert_equal(mintypecode(itype + 'D'), 'D')
        assert_equal(mintypecode('ff'), 'f')
        assert_equal(mintypecode('fd'), 'd')
        assert_equal(mintypecode('fF'), 'F')
        assert_equal(mintypecode('fD'), 'D')
        assert_equal(mintypecode('df'), 'd')
        assert_equal(mintypecode('dd'), 'd')
        assert_equal(mintypecode('dF'), 'D')
        assert_equal(mintypecode('dD'), 'D')
        assert_equal(mintypecode('Ff'), 'F')
        assert_equal(mintypecode('Fd'), 'D')
        assert_equal(mintypecode('FF'), 'F')
        assert_equal(mintypecode('FD'), 'D')
        assert_equal(mintypecode('Df'), 'D')
        assert_equal(mintypecode('Dd'), 'D')
        assert_equal(mintypecode('DF'), 'D')
        assert_equal(mintypecode('DD'), 'D')

    def test_default_3(self):
        assert_equal(mintypecode('fdF'), 'D')
        assert_equal(mintypecode('fdD'), 'D')
        assert_equal(mintypecode('fFD'), 'D')
        assert_equal(mintypecode('dFD'), 'D')
        assert_equal(mintypecode('ifd'), 'd')
        assert_equal(mintypecode('ifF'), 'F')
        assert_equal(mintypecode('ifD'), 'D')
        assert_equal(mintypecode('idF'), 'D')
        assert_equal(mintypecode('idD'), 'D')