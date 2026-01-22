from typing import Collection, Dict, List, Union, overload, Iterable
from typing_extensions import Literal
import msgpack
from pkg_resources import resource_filename
import numpy as np
from numpy.typing import ArrayLike
from .._cache import cache
from .._typing import _FloatLike_co
def __harmonic_distance(logs, a, b):
    """Compute the harmonic distance between ratios a and b.

    Harmonic distance is defined as `log2(a * b) - 2*log2(gcd(a, b))` [#]_.

    Here we are expressing a and b as prime factorization exponents,
    and the prime basis are provided in their log2 form.

    .. [#] Tenney, James.
        "On ‘Crystal Growth’ in harmonic space (1993–1998)."
        Contemporary Music Review 27.1 (2008): 47-56.
    """
    a = np.array(a)
    b = np.array(b)
    a_num = np.maximum(a, 0)
    a_den = a_num - a
    b_num = np.maximum(b, 0)
    b_den = b_num - b
    gcd = np.minimum(a_num, b_num) - np.maximum(a_den, b_den)
    return np.around(logs.dot(a + b - 2 * gcd), 6)