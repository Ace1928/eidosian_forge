import collections
import datetime
import enum
import io
import re
import struct
import typing
from spnego._text import to_text
from spnego._version import __version__ as pyspnego_version
class NegotiateFlags(enum.IntFlag):
    """NTLM Negotiation flags.

    Used during NTLM negotiation to negotiate the capabilities between the client and server.

    .. _NEGOTIATE:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/99d90ff4-957f-4c8a-80e4-5bfe5a9a9832
    """
    key_56 = 2147483648
    key_exch = 1073741824
    key_128 = 536870912
    r1 = 268435456
    r2 = 134217728
    r3 = 67108864
    version = 33554432
    r4 = 16777216
    target_info = 8388608
    non_nt_session_key = 4194304
    r5 = 2097152
    identity = 1048576
    extended_session_security = 524288
    target_type_share = 262144
    target_type_server = 131072
    target_type_domain = 65536
    always_sign = 32768
    local_call = 16384
    oem_workstation_supplied = 8192
    oem_domain_name_supplied = 4096
    anonymous = 2048
    r8 = 1024
    ntlm = 512
    r9 = 256
    lm_key = 128
    datagram = 64
    seal = 32
    sign = 16
    netware = 8
    request_target = 4
    oem = 2
    unicode = 1

    @classmethod
    def native_labels(cls) -> typing.Dict['NegotiateFlags', str]:
        return {NegotiateFlags.key_56: 'NTLMSSP_NEGOTIATE_56', NegotiateFlags.key_exch: 'NTLMSSP_NEGOTIATE_KEY_EXCH', NegotiateFlags.key_128: 'NTLMSSP_NEGOTIATE_128', NegotiateFlags.r1: 'NTLMSSP_RESERVED_R1', NegotiateFlags.r2: 'NTLMSSP_RESERVED_R2', NegotiateFlags.r3: 'NTLMSSP_RESERVED_R3', NegotiateFlags.version: 'NTLMSSP_NEGOTIATE_VERSION', NegotiateFlags.r4: 'NTLMSSP_RESERVED_R4', NegotiateFlags.target_info: 'NTLMSSP_NEGOTIATE_TARGET_INFO', NegotiateFlags.non_nt_session_key: 'NTLMSSP_REQUEST_NON_NT_SESSION_KEY', NegotiateFlags.r5: 'NTLMSSP_RESERVED_R5', NegotiateFlags.identity: 'NTLMSSP_NEGOTIATE_IDENTITY', NegotiateFlags.extended_session_security: 'NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY', NegotiateFlags.target_type_share: 'NTLMSSP_TARGET_TYPE_SHARE - R6', NegotiateFlags.target_type_server: 'NTLMSSP_TARGET_TYPE_SERVER', NegotiateFlags.target_type_domain: 'NTLMSSP_TARGET_TYPE_DOMAIN', NegotiateFlags.always_sign: 'NTLMSSP_NEGOTIATE_ALWAYS_SIGN', NegotiateFlags.local_call: 'NTLMSSP_NEGOTIATE_LOCAL_CALL - R7', NegotiateFlags.oem_workstation_supplied: 'NTLMSSP_NEGOTIATE_OEM_WORKSTATION_SUPPLIED', NegotiateFlags.oem_domain_name_supplied: 'NTLMSSP_NEGOTIATE_OEM_DOMAIN_SUPPLIED', NegotiateFlags.anonymous: 'NTLMSSP_ANOYNMOUS', NegotiateFlags.r8: 'NTLMSSP_RESERVED_R8', NegotiateFlags.ntlm: 'NTLMSSP_NEGOTIATE_NTLM', NegotiateFlags.r9: 'NTLMSSP_RESERVED_R9', NegotiateFlags.lm_key: 'NTLMSSP_NEGOTIATE_LM_KEY', NegotiateFlags.datagram: 'NTLMSSP_NEGOTIATE_DATAGRAM', NegotiateFlags.seal: 'NTLMSSP_NEGOTIATE_SEAL', NegotiateFlags.sign: 'NTLMSSP_NEGOTIATE_SIGN', NegotiateFlags.netware: 'NTLMSSP_NEGOTIATE_NETWARE - R10', NegotiateFlags.request_target: 'NTLMSSP_REQUEST_TARGET', NegotiateFlags.oem: 'NTLMSSP_NEGOTIATE_OEM', NegotiateFlags.unicode: 'NTLMSSP_NEGOTIATE_UNICODE'}