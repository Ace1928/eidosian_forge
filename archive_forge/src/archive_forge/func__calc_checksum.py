from hashlib import md5
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_unicode
from passlib.utils.binary import h64
from passlib.utils.compat import byte_elem_value, irange, u, \
import passlib.utils.handlers as uh
def _calc_checksum(self, secret):
    if isinstance(secret, unicode):
        secret = secret.encode('utf-8')
    config = str_to_bascii(self.to_string(_withchk=False))
    return raw_sun_md5_crypt(secret, self.rounds, config).decode('ascii')