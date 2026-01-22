from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
class H64Big_Test(_Base64Test):
    """test H64Big codec functions"""
    engine = h64big
    descriptionPrefix = 'h64big codec'
    encoded_data = [(b'', b''), (b'U', b'JE'), (b'U\xaa', b'JOc'), (b'U\xaaU', b'JOdJ'), (b'U\xaaU\xaa', b'JOdJeU'), (b'U\xaaU\xaaU', b'JOdJeZI'), (b'U\xaaU\xaaU\xaa', b'JOdJeZKe'), (b'U\xaaU\xaf', b'JOdJfk'), (b'U\xaaU\xaa_', b'JOdJeZw')]
    encoded_ints = [(b'.z', 63, 12), (b'z.', 4032, 12)]