import base64
import contextlib
import enum
import errno
import os
import os.path
import selectors
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple, Union
import dns._features
import dns.exception
import dns.inet
import dns.message
import dns.name
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.xfr
def _make_dot_ssl_context(server_hostname: Optional[str], verify: Union[bool, str]) -> ssl.SSLContext:
    cafile: Optional[str] = None
    capath: Optional[str] = None
    if isinstance(verify, str):
        if os.path.isfile(verify):
            cafile = verify
        elif os.path.isdir(verify):
            capath = verify
        else:
            raise ValueError('invalid verify string')
    ssl_context = ssl.create_default_context(cafile=cafile, capath=capath)
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    if server_hostname is None:
        ssl_context.check_hostname = False
    ssl_context.set_alpn_protocols(['dot'])
    if verify is False:
        ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context