import random
from mpmath import *
from mpmath.libmp import *
class mympc:

    @property
    def _mpc_(self):
        return (mpf(3.5)._mpf_, mpf(2.5)._mpf_)