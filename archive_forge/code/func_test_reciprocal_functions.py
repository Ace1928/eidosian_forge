from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_reciprocal_functions():
    assert sec(3).ae(-1.0101086659079936)
    assert csc(3).ae(7.086167395737186)
    assert cot(3).ae(-7.015252551434534)
    assert sech(3).ae(0.0993279274194332)
    assert csch(3).ae(0.09982156966882273)
    assert coth(3).ae(1.0049698233136892)
    assert asec(3).ae(1.2309594173407747)
    assert acsc(3).ae(0.3398369094541219)
    assert acot(3).ae(0.3217505543966422)
    assert asech(0.5).ae(1.3169578969248168)
    assert acsch(3).ae(0.32745015023725843)
    assert acoth(3).ae(0.34657359027997264)
    assert acot(0).ae(1.5707963267948966)
    assert acoth(0).ae(1.5707963267948966j)