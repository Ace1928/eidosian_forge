import unittest
import os
import sys
import subprocess
from collections import defaultdict
from textwrap import dedent
import numpy as np
from numba import jit, config, typed, typeof
from numba.core import types, utils
import unittest
from numba.tests.support import (TestCase, skip_unless_py10_or_later,
from numba.cpython.unicode import compile_time_get_string_data
from numba.cpython import hashing
def test_warn_on_fnv(self):
    work = '\n        import sys\n        import warnings\n        from collections import namedtuple\n\n        # hash_info is a StructSequence, mock as a named tuple\n        fields = ["width", "modulus", "inf", "nan", "imag", "algorithm",\n                  "hash_bits", "seed_bits", "cutoff"]\n\n        hinfo = sys.hash_info\n        FAKE_HASHINFO = namedtuple(\'FAKE_HASHINFO\', fields)\n\n        fd = dict()\n        for f in fields:\n            fd[f] = getattr(hinfo, f)\n\n        fd[\'algorithm\'] = \'fnv\'\n\n        fake_hashinfo = FAKE_HASHINFO(**fd)\n\n        # replace the hashinfo with the fnv version\n        sys.hash_info = fake_hashinfo\n        with warnings.catch_warnings(record=True) as warns:\n            # Cause all warnings to always be triggered.\n            warnings.simplefilter("always")\n            from numba import njit\n            @njit\n            def foo():\n                hash(1)\n            foo()\n            assert len(warns) > 0\n            expect = "FNV hashing is not implemented in Numba. See PEP 456"\n            for w in warns:\n                if expect in str(w.message):\n                    break\n            else:\n                raise RuntimeError("Expected warning not found")\n        '
    subprocess.check_call([sys.executable, '-c', dedent(work)])