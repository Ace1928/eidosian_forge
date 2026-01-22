from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class _ldap_md5_crypt_test(HandlerCase):
    handler = hash.ldap_md5_crypt
    known_correct_hashes = [('', '{CRYPT}$1$dOHYPKoP$tnxS1T8Q6VVn3kpV8cN6o.'), (' ', '{CRYPT}$1$m/5ee7ol$bZn0kIBFipq39e.KDXX8I0'), ('test', '{CRYPT}$1$ec6XvcoW$ghEtNK2U1MC5l.Dwgi3020'), ('Compl3X AlphaNu3meric', '{CRYPT}$1$nX1e7EeI$ljQn72ZUgt6Wxd9hfvHdV0'), ('4lpHa N|_|M3r1K W/ Cur5Es: #$%(*)(*%#', '{CRYPT}$1$jQS7o98J$V6iTcr71CGgwW2laf17pi1'), ('test', '{CRYPT}$1$SuMrG47N$ymvzYjr7QcEQjaK5m1PGx1'), (UPASS_TABLE, '{CRYPT}$1$d6/Ky1lU$/xpf8m7ftmWLF.TjHCqel0')]
    known_malformed_hashes = ['{CRYPT}$1$dOHYPKoP$tnxS1T8Q6VVn3kpV8cN6o!']