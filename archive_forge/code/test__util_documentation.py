from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import os
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises, deprecated_call
import scipy
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
Test that 'from numpy import *' functions are deprecated.