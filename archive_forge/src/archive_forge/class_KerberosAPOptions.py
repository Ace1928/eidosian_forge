import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosAPOptions(enum.IntFlag):
    mutual_required = 32
    use_session_key = 64
    reserved = 128

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosAPOptions', str]:
        return {KerberosAPOptions.mutual_required: 'mutual-required', KerberosAPOptions.use_session_key: 'use-session-key', KerberosAPOptions.reserved: 'reserved'}