from cupy import _core
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._zeta import zeta_definition

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/master/scipy/special/cephes/const.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igam.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igam.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/igami.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/lanczos.c
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/lanczos.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/mconf.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/polevl.h
https://github.com/scipy/scipy/blob/master/scipy/special/cephes/unity.c

Cephes Math Library Release 2.0:  April, 1987
Copyright 1984, 1987 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
