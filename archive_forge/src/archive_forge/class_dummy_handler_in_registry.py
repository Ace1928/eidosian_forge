from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
class dummy_handler_in_registry(object):
    """context manager that inserts dummy handler in registry"""

    def __init__(self, name):
        self.name = name
        self.dummy = type('dummy_' + name, (uh.GenericHandler,), dict(name=name, setting_kwds=()))

    def __enter__(self):
        from passlib import registry
        registry._unload_handler_name(self.name, locations=False)
        registry.register_crypt_handler(self.dummy)
        assert registry.get_crypt_handler(self.name) is self.dummy
        return self.dummy

    def __exit__(self, *exc_info):
        from passlib import registry
        registry._unload_handler_name(self.name, locations=False)