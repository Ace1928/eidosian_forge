from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
from passlib import apps, hash as hashmod
from passlib.tests.utils import TestCase
def test_roundup_context(self):
    ctx = apps.roundup_context
    for hash in ['{PBKDF2}9849$JMTYu3eOUSoFYExprVVqbQ$N5.gV.uR1.BTgLSvi0qyPiRlGZ0', '{SHA}a94a8fe5ccb19ba61c4c0873d391e987982fbbd3', '{CRYPT}dptOmKDriOGfU', '{plaintext}test']:
        self.assertTrue(ctx.verify('test', hash))