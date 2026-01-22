from sympy.core.random import _randint
from sympy.polys.galoistools import (
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.polyutils import _sort_factors
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
from sympy.utilities import subsets
from math import ceil as _ceil, log as _log

    Returns ``True`` if a multivariate polynomial ``f`` has no factors
    over its domain.
    