from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def _multiply_crc_polynomials(p, q):
    """Multiplies two polynomials together modulo CASTAGNOLI_POLY.

  Args:
    p (int): The first polynomial.
    q (int): The second polynomial.

  Returns:
    Int result of the multiplication.
  """
    result = 0
    top_bit = 1 << DEGREE
    for _ in range(DEGREE):
        if p & 1:
            result ^= q
        q <<= 1
        if q & top_bit:
            q ^= CASTAGNOLI_POLY
        p >>= 1
    return result