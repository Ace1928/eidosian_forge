from hashlib import md5
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, repeat_string
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u
import passlib.utils.handlers as uh
@classmethod
def _load_backend_os_crypt(cls):
    if test_crypt('test', '$1$test$pi/xDtU5WFVRqYS6BMU8X/'):
        cls._set_calc_checksum_backend(cls._calc_checksum_os_crypt)
        return True
    else:
        return False