import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def _create_ssl_ctx(self, sslp):
    if isinstance(sslp, ssl.SSLContext):
        return sslp
    ca = sslp.get('ca')
    capath = sslp.get('capath')
    hasnoca = ca is None and capath is None
    ctx = ssl.create_default_context(cafile=ca, capath=capath)
    ctx.check_hostname = not hasnoca and sslp.get('check_hostname', True)
    verify_mode_value = sslp.get('verify_mode')
    if verify_mode_value is None:
        ctx.verify_mode = ssl.CERT_NONE if hasnoca else ssl.CERT_REQUIRED
    elif isinstance(verify_mode_value, bool):
        ctx.verify_mode = ssl.CERT_REQUIRED if verify_mode_value else ssl.CERT_NONE
    else:
        if isinstance(verify_mode_value, str):
            verify_mode_value = verify_mode_value.lower()
        if verify_mode_value in ('none', '0', 'false', 'no'):
            ctx.verify_mode = ssl.CERT_NONE
        elif verify_mode_value == 'optional':
            ctx.verify_mode = ssl.CERT_OPTIONAL
        elif verify_mode_value in ('required', '1', 'true', 'yes'):
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx.verify_mode = ssl.CERT_NONE if hasnoca else ssl.CERT_REQUIRED
    if 'cert' in sslp:
        ctx.load_cert_chain(sslp['cert'], keyfile=sslp.get('key'))
    if 'cipher' in sslp:
        ctx.set_ciphers(sslp['cipher'])
    ctx.options |= ssl.OP_NO_SSLv2
    ctx.options |= ssl.OP_NO_SSLv3
    return ctx