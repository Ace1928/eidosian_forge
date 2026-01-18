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
def test_parse_header_trailing_newline(self):
    c = Commit.from_string(b'tree a7d6277f78d3ecd0230a1a5df6db00b1d9c521ac\nparent c09b6dec7a73760fbdb478383a3c926b18db8bbe\nauthor Neil Matatall <oreoshake@github.com> 1461964057 -1000\ncommitter Neil Matatall <oreoshake@github.com> 1461964057 -1000\ngpgsig -----BEGIN PGP SIGNATURE-----\n \n wsBcBAABCAAQBQJXI80ZCRA6pcNDcVZ70gAAarcIABs72xRX3FWeox349nh6ucJK\n CtwmBTusez2Zwmq895fQEbZK7jpaGO5TRO4OvjFxlRo0E08UFx3pxZHSpj6bsFeL\n hHsDXnCaotphLkbgKKRdGZo7tDqM84wuEDlh4MwNe7qlFC7bYLDyysc81ZX5lpMm\n 2MFF1TvjLAzSvkT7H1LPkuR3hSvfCYhikbPOUNnKOo0sYjeJeAJ/JdAVQ4mdJIM0\n gl3REp9+A+qBEpNQI7z94Pg5Bc5xenwuDh3SJgHvJV6zBWupWcdB3fAkVd4TPnEZ\n nHxksHfeNln9RKseIDcy4b2ATjhDNIJZARHNfr6oy4u3XPW4svRqtBsLoMiIeuI=\n =ms6q\n -----END PGP SIGNATURE-----\n \n\n3.3.0 version bump and docs\n')
    self.assertEqual([], c._extra)
    self.assertEqual(b'-----BEGIN PGP SIGNATURE-----\n\nwsBcBAABCAAQBQJXI80ZCRA6pcNDcVZ70gAAarcIABs72xRX3FWeox349nh6ucJK\nCtwmBTusez2Zwmq895fQEbZK7jpaGO5TRO4OvjFxlRo0E08UFx3pxZHSpj6bsFeL\nhHsDXnCaotphLkbgKKRdGZo7tDqM84wuEDlh4MwNe7qlFC7bYLDyysc81ZX5lpMm\n2MFF1TvjLAzSvkT7H1LPkuR3hSvfCYhikbPOUNnKOo0sYjeJeAJ/JdAVQ4mdJIM0\ngl3REp9+A+qBEpNQI7z94Pg5Bc5xenwuDh3SJgHvJV6zBWupWcdB3fAkVd4TPnEZ\nnHxksHfeNln9RKseIDcy4b2ATjhDNIJZARHNfr6oy4u3XPW4svRqtBsLoMiIeuI=\n=ms6q\n-----END PGP SIGNATURE-----\n', c.gpgsig)