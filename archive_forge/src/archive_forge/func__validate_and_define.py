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
def _validate_and_define(params, key, value):
    key, force_generic = _validate_key(_unescape(key))
    if key in params:
        raise SyntaxError(f'duplicate key "{key:d}"')
    cls = _class_for_key.get(key, GenericParam)
    emptiness = cls.emptiness()
    if value is None:
        if emptiness == Emptiness.NEVER:
            raise SyntaxError('value cannot be empty')
        value = cls.from_value(value)
    elif force_generic:
        value = cls.from_wire_parser(dns.wire.Parser(_unescape(value)))
    else:
        value = cls.from_value(value)
    params[key] = value