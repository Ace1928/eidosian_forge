import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def test_read_tag_from_file(self):
    t = self.get_tag(tag_sha)
    self.assertEqual(t.object, (Commit, b'51b668fd5bf7061b7d6fa525f88803e6cfadaa51'))
    self.assertEqual(t.name, b'signed')
    self.assertEqual(t.tagger, b'Ali Sabil <ali.sabil@gmail.com>')
    self.assertEqual(t.tag_time, 1231203091)
    self.assertEqual(t.message, b'This is a signed tag\n')
    self.assertEqual(t.signature, b'-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.9 (GNU/Linux)\n\niEYEABECAAYFAkliqx8ACgkQqSMmLy9u/kcx5ACfakZ9NnPl02tOyYP6pkBoEkU1\n5EcAn0UFgokaSvS371Ym/4W9iJj6vh3h\n=ql7y\n-----END PGP SIGNATURE-----\n')