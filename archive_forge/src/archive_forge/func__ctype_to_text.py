import struct
import base64
import dns.exception
import dns.dnssec
import dns.rdata
import dns.tokenizer
def _ctype_to_text(what):
    v = _ctype_by_value.get(what)
    if v is not None:
        return v
    return str(what)