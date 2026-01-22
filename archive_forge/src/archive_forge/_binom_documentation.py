from cupy import _core
from cupyx.scipy.special._beta import (
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._gammainc import p1evl_definition

Implements the binom function from scipy.

This is essentially a CUDA C++ adaptation of existing scipy code, available at:
https://github.com/scipy/scipy/blob/v1.10.1/scipy/special/orthogonal_eval.pxd
