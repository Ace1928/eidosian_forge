import sys, os
from timeit import default_timer as clock
from mpmath import *
from mpmath.libmp.backend import exec_

Torture tests for asymptotics and high precision evaluation of
special functions.

(Other torture tests may also be placed here.)

Running this file (gmpy recommended!) takes several CPU minutes.
With Python 2.6+, multiprocessing is used automatically to run tests
in parallel if many cores are available. (A single test may take between
a second and several minutes; possibly more.)

The idea:

* We evaluate functions at positive, negative, imaginary, 45- and 135-degree
  complex values with magnitudes between 10^-20 to 10^20, at precisions between
  5 and 150 digits (we can go even higher for fast functions).

* Comparing the result from two different precision levels provides
  a strong consistency check (particularly for functions that use
  different algorithms at different precision levels).

* That the computation finishes at all (without failure), within reasonable
  time, provides a check that evaluation works at all: that the code runs,
  that it doesn't get stuck in an infinite loop, and that it doesn't use
  some extremely slowly algorithm where it could use a faster one.

TODO:

* Speed up those functions that take long to finish!
* Generalize to test more cases; more options.
* Implement a timeout mechanism.
* Some functions are notably absent, including the following:
  * inverse trigonometric functions (some become inaccurate for complex arguments)
  * ci, si (not implemented properly for large complex arguments)
  * zeta functions (need to modify test not to try too large imaginary values)
  * and others...

