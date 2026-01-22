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
class TestHashAlgs(TestCase):
    known_hashes = {'djba33x': [[193485960, 193485960, 193485960, 193485960], [-678966196, 573763426263223372, -820489388, -4282905804826039665]], 'siphash13': [[69611762, -4594863902769663758, 69611762, -4594863902769663758], [-975800855, 3869580338025362921, -975800855, 3869580338025362921], [-595844228, 7764564197781545852, -595844228, 7764564197781545852], [-1093288643, -2810468059467891395, -1041341092, 4925090034378237276], [-585999602, -2845126246016066802, -817336969, -2219421378907968137]], 'siphash24': [[1198583518, 4596069200710135518, 1198583518, 4596069200710135518], [273876886, -4501618152524544106, 273876886, -4501618152524544106], [-1745215313, 4436719588892876975, -1745215313, 4436719588892876975], [493570806, 5749986484189612790, -1006381564, -5915111450199468540], [-1677110816, -2947981342227738144, -1860207793, -4296699217652516017]]}

    def get_expected_hash(self, position, length):
        if length < sys.hash_info.cutoff:
            algorithm = 'djba33x'
        else:
            algorithm = sys.hash_info.algorithm
        IS_64BIT = not config.IS_32BITS
        if sys.byteorder == 'little':
            platform = 1 if IS_64BIT else 0
        else:
            assert sys.byteorder == 'big'
            platform = 3 if IS_64BIT else 2
        return self.known_hashes[algorithm][position][platform]

    def get_hash_command(self, repr_):
        return 'print(hash(eval(%a)))' % repr_

    def get_hash(self, repr_, seed=None):
        env = os.environ.copy()
        if seed is not None:
            env['PYTHONHASHSEED'] = str(seed)
        else:
            env.pop('PYTHONHASHSEED', None)
        out, _ = run_in_subprocess(code=self.get_hash_command(repr_), env=env)
        stdout = out.decode().strip()
        return int(stdout)

    def test_against_cpython_gold(self):
        args = (('abc', 0, 0), ('abc', 42, 1), ('abcdefghijk', 42, 2), ('äú∑ℇ', 0, 3), ('äú∑ℇ', 42, 4))
        for input_str, seed, position in args:
            with self.subTest(input_str=input_str, seed=seed):
                got = self.get_hash(repr(input_str), seed=seed)
                expected = self.get_expected_hash(position, len(input_str))
                self.assertEqual(got, expected)