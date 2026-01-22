from hashlib import md5
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, repeat_string
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u
import passlib.utils.handlers as uh
def _raw_md5_crypt(pwd, salt, use_apr=False):
    """perform raw md5-crypt calculation

    this function provides a pure-python implementation of the internals
    for the MD5-Crypt algorithms; it doesn't handle any of the
    parsing/validation of the hash strings themselves.

    :arg pwd: password chars/bytes to hash
    :arg salt: salt chars to use
    :arg use_apr: use apache variant

    :returns:
        encoded checksum chars
    """
    if isinstance(pwd, unicode):
        pwd = pwd.encode('utf-8')
    assert isinstance(pwd, bytes), 'pwd not unicode or bytes'
    if _BNULL in pwd:
        raise uh.exc.NullPasswordError(md5_crypt)
    pwd_len = len(pwd)
    assert isinstance(salt, unicode), 'salt not unicode'
    salt = salt.encode('ascii')
    assert len(salt) < 9, 'salt too large'
    if use_apr:
        magic = _APR_MAGIC
    else:
        magic = _MD5_MAGIC
    db = md5(pwd + salt + pwd).digest()
    a_ctx = md5(pwd + magic + salt)
    a_ctx_update = a_ctx.update
    a_ctx_update(repeat_string(db, pwd_len))
    i = pwd_len
    evenchar = pwd[:1]
    while i:
        a_ctx_update(_BNULL if i & 1 else evenchar)
        i >>= 1
    da = a_ctx.digest()
    pwd_pwd = pwd + pwd
    pwd_salt = pwd + salt
    perms = [pwd, pwd_pwd, pwd_salt, pwd_salt + pwd, salt + pwd, salt + pwd_pwd]
    data = [(perms[even], perms[odd]) for even, odd in _c_digest_offsets]
    dc = da
    blocks = 23
    while blocks:
        for even, odd in data:
            dc = md5(odd + md5(dc + even).digest()).digest()
        blocks -= 1
    for even, odd in data[:17]:
        dc = md5(odd + md5(dc + even).digest()).digest()
    return h64.encode_transposed_bytes(dc, _transpose_map).decode('ascii')