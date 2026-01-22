import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _build_tls_unknown_ca_alert(version):
    """
    Builds a TLS alert record for an unknown CA.
    """
    ver_maj, ver_min = TLS_PROTOCOL_VERSIONS[version]
    severity_fatal = 2
    description_unknown_ca = 48
    msg = struct.pack('>BB', severity_fatal, description_unknown_ca)
    msg_len = len(msg)
    record_type_alert = 21
    record = struct.pack('>BBBH', record_type_alert, ver_maj, ver_min, msg_len) + msg
    return record