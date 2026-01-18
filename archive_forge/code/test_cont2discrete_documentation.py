import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep

        Test that the solution to the discrete approximation of a continuous
        system actually approximates the solution to the continuous system.
        This is an indirect test of the correctness of the implementation
        of cont2discrete.
        