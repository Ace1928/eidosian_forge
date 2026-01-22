from cupy import _core
from cupyx.scipy.special._beta import incbet_preamble, incbi_preamble
from cupyx.scipy.special._gammainc import _igam_preamble, _igami_preamble
Statistical distribution functions (Beta, Binomial, Poisson, etc.)

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/bdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/chdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/fdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/gdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/nbdtr.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/pdtr.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
