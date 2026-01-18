import random
from mpmath import *
from mpmath.libmp import *
def test_tight_string_conversion():
    mp.dps = 15
    assert from_str('0.5', 10, round_floor) == fhalf
    assert from_str('0.5', 10, round_ceiling) == fhalf