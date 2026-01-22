from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import re
import warnings
from passlib import hash
from passlib.utils import repeat_string
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, SkipTest
from passlib.tests.test_handlers import UPASS_USD, UPASS_TABLE
from passlib.tests.test_ext_django import DJANGO_VERSION, MIN_DJANGO_VERSION, \
from passlib.tests.test_handlers_argon2 import _base_argon2_test
@skipUnless(hash.argon2.has_backend(), 'no argon2 backends available')
class django_argon2_test(HandlerCase, _DjangoHelper):
    """test django_bcrypt"""
    handler = hash.django_argon2
    known_correct_hashes = [('password', 'argon2$argon2i$v=19$m=256,t=1,p=1$c29tZXNhbHQ$AJFIsNZTMKTAewB4+ETN1A'), ('password', 'argon2$argon2i$v=19$m=380,t=2,p=2$c29tZXNhbHQ$SrssP8n7m/12VWPM8dvNrw'), (UPASS_LETMEIN, 'argon2$argon2i$v=19$m=512,t=2,p=2$V25jN1l4UUJZWkR1$MxpA1BD2Gh7+D79gaAw6sQ')]

    def setUpWarnings(self):
        super(django_argon2_test, self).setUpWarnings()
        warnings.filterwarnings('ignore', '.*Using argon2pure backend.*')

    def do_stub_encrypt(self, handler=None, **settings):
        handler = (handler or self.handler).using(**settings)
        self = handler.wrapped(use_defaults=True)
        self.checksum = self._stub_checksum
        assert self.checksum
        return handler._wrap_hash(self.to_string())

    def test_03_legacy_hash_workflow(self):
        raise self.skipTest('legacy 1.6 workflow not supported')

    class FuzzHashGenerator(_base_argon2_test.FuzzHashGenerator):

        def random_type(self):
            return 'I'

        def random_rounds(self):
            return self.randintgauss(1, 3, 2, 1)