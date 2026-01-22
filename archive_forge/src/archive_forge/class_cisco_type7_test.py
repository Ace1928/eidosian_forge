from __future__ import absolute_import, division, print_function
import logging
from passlib import hash, exc
from passlib.utils.compat import u
from .utils import UserHandlerMixin, HandlerCase, repeat_string
from .test_handlers import UPASS_TABLE
class cisco_type7_test(HandlerCase):
    handler = hash.cisco_type7
    salt_bits = 4
    salt_type = int
    known_correct_hashes = [('secure ', '04480E051A33490E'), ('Its time to go to lunch!', '153B1F1F443E22292D73212D5300194315591954465A0D0B59'), ('t35t:pa55w0rd', '08351F1B1D431516475E1B54382F'), ('hiImTesting:)', '020E0D7206320A325847071E5F5E'), ('cisco123', '060506324F41584B56'), ('cisco123', '1511021F07257A767B'), ('Supe&8ZUbeRp4SS', '06351A3149085123301517391C501918'), (UPASS_TABLE, '0958EDC8A9F495F6F8A5FD')]
    known_unidentified_hashes = ['0A480E051A33490E', '99400E4812']

    def test_90_decode(self):
        """test cisco_type7.decode()"""
        from passlib.utils import to_unicode, to_bytes
        handler = self.handler
        for secret, hash in self.known_correct_hashes:
            usecret = to_unicode(secret)
            bsecret = to_bytes(secret)
            self.assertEqual(handler.decode(hash), usecret)
            self.assertEqual(handler.decode(hash, None), bsecret)
        self.assertRaises(UnicodeDecodeError, handler.decode, '0958EDC8A9F495F6F8A5FD', 'ascii')

    def test_91_salt(self):
        """test salt value border cases"""
        handler = self.handler
        self.assertRaises(TypeError, handler, salt=None)
        handler(salt=None, use_defaults=True)
        self.assertRaises(TypeError, handler, salt='abc')
        self.assertRaises(ValueError, handler, salt=-10)
        self.assertRaises(ValueError, handler, salt=100)
        self.assertRaises(TypeError, handler.using, salt='abc')
        self.assertRaises(ValueError, handler.using, salt=-10)
        self.assertRaises(ValueError, handler.using, salt=100)
        with self.assertWarningList('salt/offset must be.*'):
            subcls = handler.using(salt=100, relaxed=True)
        self.assertEqual(subcls(use_defaults=True).salt, 52)