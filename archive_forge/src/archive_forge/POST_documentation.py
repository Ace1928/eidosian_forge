from numba.tests import test_runtests
from numba import njit
 Numba's POWER ON SELF TEST script. Used by CI to check:
0. That Numba imports ok!
1. That Numba can find an appropriate number of its own tests to run.
2. That Numba can manage to correctly compile and execute at least one thing.
