import base64
import enum
import struct
import dns.enum
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
import dns.rdtypes.util
import dns.renderer
import dns.tokenizer
import dns.wire
class ParamKey(dns.enum.IntEnum):
    """SVCB ParamKey"""
    MANDATORY = 0
    ALPN = 1
    NO_DEFAULT_ALPN = 2
    PORT = 3
    IPV4HINT = 4
    ECH = 5
    IPV6HINT = 6
    DOHPATH = 7

    @classmethod
    def _maximum(cls):
        return 65535

    @classmethod
    def _short_name(cls):
        return 'SVCBParamKey'

    @classmethod
    def _prefix(cls):
        return 'KEY'

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownParamKey