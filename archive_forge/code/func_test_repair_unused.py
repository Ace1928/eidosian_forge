from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
def test_repair_unused(self):
    """test repair_unused()"""
    from passlib.utils import getrandstr
    rng = self.getRandom()
    engine = self.engine
    check_repair_unused = self.engine.check_repair_unused
    i = 0
    while i < 300:
        size = rng.randint(0, 23)
        cdata = getrandstr(rng, engine.charmap, size).encode('ascii')
        if size & 3 == 1:
            self.assertRaises(ValueError, check_repair_unused, cdata)
            continue
        rdata = engine.encode_bytes(engine.decode_bytes(cdata))
        if rng.random() < 0.5:
            cdata = cdata.decode('ascii')
            rdata = rdata.decode('ascii')
        if cdata == rdata:
            ok, result = check_repair_unused(cdata)
            self.assertFalse(ok)
            self.assertEqual(result, rdata)
        else:
            self.assertNotEqual(size % 4, 0)
            ok, result = check_repair_unused(cdata)
            self.assertTrue(ok)
            self.assertEqual(result, rdata)
        i += 1