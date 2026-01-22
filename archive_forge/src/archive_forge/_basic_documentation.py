from cupy import _core
basic special functions


cotdg and tandg implementations are adapted from the following SciPy code:

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/tandg.c

radian is from

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/sindg.c

cosm1 is from

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/unity.c

polevl is from

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/polevl.h


Cephes Math Library Release 2.0:  April, 1987
Copyright 1984, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
