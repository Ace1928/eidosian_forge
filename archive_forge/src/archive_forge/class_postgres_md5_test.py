from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class postgres_md5_test(UserHandlerMixin, HandlerCase):
    handler = hash.postgres_md5
    known_correct_hashes = [(('mypass', 'postgres'), 'md55fba2ea04fd36069d2574ea71c8efe9d'), (('mypass', 'root'), 'md540c31989b20437833f697e485811254b'), (('testpassword', 'testuser'), 'md5d4fc5129cc2c25465a5370113ae9835f'), ((UPASS_TABLE, 'postgres'), 'md5cb9f11283265811ce076db86d18a22d2')]
    known_unidentified_hashes = ['md54zc31989b20437833f697e485811254b']