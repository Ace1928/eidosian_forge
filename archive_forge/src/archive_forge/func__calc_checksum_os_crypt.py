from hashlib import md5
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, repeat_string
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u
import passlib.utils.handlers as uh
def _calc_checksum_os_crypt(self, secret):
    config = self.ident + self.salt
    hash = safe_crypt(secret, config)
    if hash is None:
        return self._calc_checksum_builtin(secret)
    if not hash.startswith(config) or len(hash) != len(config) + 23:
        raise uh.exc.CryptBackendError(self, config, hash)
    return hash[-22:]