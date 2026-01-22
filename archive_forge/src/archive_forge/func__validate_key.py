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
def _validate_key(key):
    force_generic = False
    if isinstance(key, bytes):
        key = key.decode('latin-1')
    if isinstance(key, str):
        if key.lower().startswith('key'):
            force_generic = True
            if key[3:].startswith('0') and len(key) != 4:
                raise ValueError('leading zeros in key')
        key = key.replace('-', '_')
    return (ParamKey.make(key), force_generic)