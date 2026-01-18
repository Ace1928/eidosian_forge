import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_shared_range():
    for ft in sctypes['float']:
        for it in sctypes['int'] + sctypes['uint']:
            mn, mx = shared_range(ft, it)
            with suppress_warnings():
                ovs = ft(mx) + np.arange(2048, dtype=ft)
            bit_bigger = ovs[np.isfinite(ovs)].astype(it)
            casted_mx = ft(mx).astype(it)
            imax = int(np.iinfo(it).max)
            thresh_overflow = False
            if casted_mx != imax:
                fimax = ft(imax)
                if np.isfinite(fimax):
                    assert int(fimax) != imax
                imax_roundtrip = fimax.astype(it)
                if imax_roundtrip == imax:
                    thresh_overflow = True
            if thresh_overflow:
                assert np.all((bit_bigger == casted_mx) | (bit_bigger == imax))
            else:
                assert np.all(bit_bigger <= casted_mx)
            if it in sctypes['uint']:
                assert mn == 0
                continue
            with suppress_warnings():
                ovs = ft(mn) - np.arange(2048, dtype=ft)
            bit_smaller = ovs[np.isfinite(ovs)].astype(it)
            casted_mn = ft(mn).astype(it)
            imin = int(np.iinfo(it).min)
            if casted_mn != imin:
                fimin = ft(imin)
                if np.isfinite(fimin):
                    assert int(fimin) != imin
                imin_roundtrip = fimin.astype(it)
                if imin_roundtrip == imin:
                    thresh_overflow = True
            if thresh_overflow:
                assert np.all((bit_smaller == casted_mn) | (bit_smaller == imin))
            else:
                assert np.all(bit_smaller >= casted_mn)