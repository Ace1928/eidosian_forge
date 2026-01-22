from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class sun_md5_crypt_test(HandlerCase):
    handler = hash.sun_md5_crypt
    known_correct_hashes = [('Gpcs3_adm', '$md5$zrdhpMlZ$$wBvMOEqbSjU.hu5T2VEP01'), ('aa12345678', '$md5$vyy8.OVF$$FY4TWzuauRl4.VQNobqMY.'), ('this', '$md5$3UqYqndY$$6P.aaWOoucxxq.l00SS9k0'), ('passwd', '$md5$RPgLF6IJ$WTvAlUJ7MqH5xak2FMEwS/'), (UPASS_TABLE, '$md5,rounds=5000$10VYDzAA$$1arAVtMA3trgE1qJ2V0Ez1')]
    known_correct_configs = [('$md5$3UqYqndY$', 'this', '$md5$3UqYqndY$$6P.aaWOoucxxq.l00SS9k0'), ('$md5$3UqYqndY$$.................DUMMY', 'this', '$md5$3UqYqndY$$6P.aaWOoucxxq.l00SS9k0'), ('$md5$3UqYqndY', 'this', '$md5$3UqYqndY$HIZVnfJNGCPbDZ9nIRSgP1'), ('$md5$3UqYqndY$.................DUMMY', 'this', '$md5$3UqYqndY$HIZVnfJNGCPbDZ9nIRSgP1')]
    known_malformed_hashes = ['$md5,rounds=5000', '$md5,rounds=500A$xxxx', '$md5,rounds=0500$xxxx', '$md5,rounds=0$xxxx', '$md5$RPgL!6IJ$WTvAlUJ7MqH5xak2FMEwS/', '$md5$RPgLa6IJ$WTvAlUJ7MqH5xak2FMEwS', '$md5$RPgLa6IJ$WTvAlUJ7MqH5xak2FMEwS/.', '$md5$3UqYqndY$$', '$md5$RPgLa6IJ$$$WTvAlUJ7MqH5xak2FMEwS/']
    platform_crypt_support = [('solaris', True), ('freebsd|openbsd|netbsd|linux|darwin', False)]

    def do_verify(self, secret, hash):
        if isinstance(hash, str) and hash.endswith('$.................DUMMY'):
            raise ValueError("pretending '$...' stub hash is config string")
        return self.handler.verify(secret, hash)