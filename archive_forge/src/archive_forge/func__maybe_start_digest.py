import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
def _maybe_start_digest(key, mac, multi):
    """If this is the first message in a multi-message sequence,
    start a new context.
    @rtype: dns.tsig.HMACTSig or dns.tsig.GSSTSig object
    """
    if multi:
        ctx = get_context(key)
        ctx.update(struct.pack('!H', len(mac)))
        ctx.update(mac)
        return ctx
    else:
        return None