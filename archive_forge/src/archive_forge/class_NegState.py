import enum
import struct
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import KerberosV5Msg
from spnego._ntlm_raw.messages import NTLMMessage
class NegState(enum.IntEnum):
    accept_complete = 0
    accept_incomplete = 1
    reject = 2
    request_mic = 3

    @classmethod
    def native_labels(cls) -> typing.Dict['NegState', str]:
        return {NegState.accept_complete: 'accept-complete', NegState.accept_incomplete: 'accept-incomplete', NegState.reject: 'reject', NegState.request_mic: 'request-mic'}