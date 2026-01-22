from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
class TestPassword(unittest.TestCase):
    """Test basic password functionality"""

    def clstest(self, cls):
        """Insure that password.__eq__ hashes test value before compare."""
        password = cls('foo')
        self.assertNotEquals(password, 'foo')
        password.set('foo')
        hashed = str(password)
        self.assertEquals(password, 'foo')
        self.assertEquals(password.str, hashed)
        password = cls(hashed)
        self.assertNotEquals(password.str, 'foo')
        self.assertEquals(password, 'foo')
        self.assertEquals(password.str, hashed)

    def test_aaa_version_1_9_default_behavior(self):
        self.clstest(Password)

    def test_custom_hashclass(self):

        class SHA224Password(Password):
            hashfunc = hashlib.sha224
        password = SHA224Password()
        password.set('foo')
        self.assertEquals(hashlib.sha224(b'foo').hexdigest(), str(password))

    def test_hmac(self):

        def hmac_hashfunc(cls, msg):
            if not isinstance(msg, bytes):
                msg = msg.encode('utf-8')
            return hmac.new(b'mysecretkey', msg)

        class HMACPassword(Password):
            hashfunc = hmac_hashfunc
        self.clstest(HMACPassword)
        password = HMACPassword()
        password.set('foo')
        self.assertEquals(str(password), hmac.new(b'mysecretkey', b'foo').hexdigest())

    def test_constructor(self):
        hmac_hashfunc = lambda msg: hmac.new(b'mysecretkey', msg)
        password = Password(hashfunc=hmac_hashfunc)
        password.set('foo')
        self.assertEquals(password.str, hmac.new(b'mysecretkey', b'foo').hexdigest())