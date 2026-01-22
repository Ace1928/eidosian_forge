import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class AvFlags(enum.IntFlag):
    """MsvAvFlags for an AV_PAIR.

    These are the flags that can be set on the MsvAvFlags entry of an NTLM `AV_PAIR`_.

    .. _AV_PAIR:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/83f5e789-660d-4781-8491-5f8c6641f75e
    """
    none = 0
    constrained = 1
    mic = 2
    untrusted_spn = 4

    @classmethod
    def native_labels(cls) -> typing.Dict['AvFlags', str]:
        return {AvFlags.constrained: 'AUTHENTICATION_CONSTRAINED', AvFlags.mic: 'MIC_PROVIDED', AvFlags.untrusted_spn: 'UNTRUSTED_SPN_SOURCE'}