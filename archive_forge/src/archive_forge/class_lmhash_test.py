from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class lmhash_test(EncodingHandlerMixin, HandlerCase):
    handler = hash.lmhash
    secret_case_insensitive = True
    known_correct_hashes = [('OLDPASSWORD', 'c9b81d939d6fd80cd408e6b105741864'), ('NEWPASSWORD', '09eeab5aa415d6e4d408e6b105741864'), ('welcome', 'c23413a8a1e7665faad3b435b51404ee'), ('', 'aad3b435b51404eeaad3b435b51404ee'), ('zzZZZzz', 'a5e6066de61c3e35aad3b435b51404ee'), ('passphrase', '855c3697d9979e78ac404c4ba2c66533'), ('Yokohama', '5ecd9236d21095ce7584248b8d2c9f9e'), (u('ENCYCLOPÆDIA'), 'fed6416bffc9750d48462b9d7aaac065'), (u('encyclopædia'), 'fed6416bffc9750d48462b9d7aaac065'), ((u('Æ'), None), '25d8ab4a0659c97aaad3b435b51404ee'), ((u('Æ'), 'cp437'), '25d8ab4a0659c97aaad3b435b51404ee'), ((u('Æ'), 'latin-1'), '184eecbbe9991b44aad3b435b51404ee'), ((u('Æ'), 'utf-8'), '00dd240fcfab20b8aad3b435b51404ee')]
    known_unidentified_hashes = ['855c3697d9979e78ac404c4ba2c6653X']

    def test_90_raw(self):
        """test lmhash.raw() method"""
        from binascii import unhexlify
        from passlib.utils.compat import str_to_bascii
        lmhash = self.handler
        for secret, hash in self.known_correct_hashes:
            kwds = {}
            secret = self.populate_context(secret, kwds)
            data = unhexlify(str_to_bascii(hash))
            self.assertEqual(lmhash.raw(secret, **kwds), data)
        self.assertRaises(TypeError, lmhash.raw, 1)