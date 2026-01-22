from base64 import b64encode, b64decode
from hashlib import md5, sha1, sha256, sha512
import logging; log = logging.getLogger(__name__)
import re
from passlib.handlers.misc import plaintext
from passlib.utils import unix_crypt_schemes, to_unicode
from passlib.utils.compat import uascii_to_str, unicode, u
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
class ldap_md5(_Base64DigestHelper):
    """This class stores passwords using LDAP's plain MD5 format, and follows the :ref:`password-hash-api`.

    The :meth:`~passlib.ifc.PasswordHash.hash` and :meth:`~passlib.ifc.PasswordHash.genconfig` methods have no optional keywords.
    """
    name = 'ldap_md5'
    ident = u('{MD5}')
    _hash_func = md5
    _hash_regex = re.compile(u('^\\{MD5\\}(?P<chk>[+/a-zA-Z0-9]{22}==)$'))