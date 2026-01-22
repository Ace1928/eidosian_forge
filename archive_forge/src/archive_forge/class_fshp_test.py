from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class fshp_test(HandlerCase):
    """test fshp algorithm"""
    handler = hash.fshp
    known_correct_hashes = [('test', '{FSHP0|0|1}qUqP5cyxm6YcTAhz05Hph5gvu9M='), ('test', '{FSHP1|8|4096}MTIzNDU2NzjTdHcmoXwNc0ff9+ArUHoN0CvlbPZpxFi1C6RDM/MHSA=='), ('OrpheanBeholderScryDoubt', '{FSHP1|8|4096}GVSUFDAjdh0vBosn1GUhzGLHP7BmkbCZVH/3TQqGIjADXpc+6NCg3g=='), ('ExecuteOrder66', '{FSHP3|16|8192}0aY7rZQ+/PR+Rd5/I9ssRM7cjguyT8ibypNaSp/U1uziNO3BVlg5qPUng+zHUDQC3ao/JbzOnIBUtAeWHEy7a2vZeZ7jAwyJJa2EqOsq4Io='), (UPASS_TABLE, '{FSHP1|16|16384}9v6/l3Lu/d9by5nznpOScqQo8eKu/b/CKli3RCkgYg4nRTgZu5y659YV8cCZ68UL')]
    known_unidentified_hashes = ['{FSHX0|0|1}qUqP5cyxm6YcTAhz05Hph5gvu9M=', 'FSHP0|0|1}qUqP5cyxm6YcTAhz05Hph5gvu9M=']
    known_malformed_hashes = ['{FSHP0|0|1}qUqP5cyxm6YcTAhz05Hph5gvu9M', '{FSHP0|1|1}qUqP5cyxm6YcTAhz05Hph5gvu9M=', '{FSHP0|0|A}qUqP5cyxm6YcTAhz05Hph5gvu9M=']

    def test_90_variant(self):
        """test variant keyword"""
        handler = self.handler
        kwds = dict(salt=b'a', rounds=1)
        handler(variant=1, **kwds)
        handler(variant=u('1'), **kwds)
        handler(variant=b'1', **kwds)
        handler(variant=u('sha256'), **kwds)
        handler(variant=b'sha256', **kwds)
        self.assertRaises(TypeError, handler, variant=None, **kwds)
        self.assertRaises(TypeError, handler, variant=complex(1, 1), **kwds)
        self.assertRaises(ValueError, handler, variant='9', **kwds)
        self.assertRaises(ValueError, handler, variant=9, **kwds)