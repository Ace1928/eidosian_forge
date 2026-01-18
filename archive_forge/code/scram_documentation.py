import logging; log = logging.getLogger(__name__)
from passlib.utils import consteq, saslprep, to_native_str, splitcomma
from passlib.utils.binary import ab64_decode, ab64_encode
from passlib.utils.compat import bascii_to_str, iteritems, u, native_string_types
from passlib.crypto.digest import pbkdf2_hmac, norm_hash_name
import passlib.utils.handlers as uh
normalize algs parameter