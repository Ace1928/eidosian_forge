from .ctx_base import StandardBaseContext
import math
import cmath
from . import math2
from . import function_docs
from .libmp import mpf_bernoulli, to_float, int_types
from . import libmp

    Context for fast low-precision arithmetic (53-bit precision, giving at most
    about 15-digit accuracy), using Python's builtin float and complex.
    