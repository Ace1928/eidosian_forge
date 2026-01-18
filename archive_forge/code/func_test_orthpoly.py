import math
import pytest
from mpmath import *
def test_orthpoly():
    mp.dps = 15
    assert jacobi(-4, 2, 3, 0.7).ae(22800.0 / 4913)
    assert jacobi(3, 2, 4, 5.5) == 4133.125
    assert jacobi(1.5, 5 / 6.0, 4, 0).ae(-1.0851951434075509)
    assert jacobi(-2, 1, 2, 4).ae(-0.16)
    assert jacobi(2, -1, 2.5, 4).ae(34.59375)
    assert legendre(5, 7) == 129367
    assert legendre(0.5, 0).ae(0.5393526011883794)
    assert legendre(-1, -1) == 1
    assert legendre(0, -1) == 1
    assert legendre(0, 1) == 1
    assert legendre(1, -1) == -1
    assert legendre(7, 1) == 1
    assert legendre(7, -1) == -1
    assert legendre(8, 1.5).ae(15457523.0 / 32768)
    assert legendre(j, -j).ae(2.4448182735671433 + 0.6928881737669935j)
    assert chebyu(5, 1) == 6
    assert chebyt(3, 2) == 26
    assert legendre(3.5, -1) == inf
    assert legendre(4.5, -1) == -inf
    assert legendre(3.5 + 1j, -1) == mpc(inf, inf)
    assert legendre(4.5 + 1j, -1) == mpc(-inf, -inf)
    assert laguerre(4, -2, 3).ae(-1.125)
    assert laguerre(3, 1 + j, 0.5).ae(0.22916666666666666 + 2.5416666666666665j)