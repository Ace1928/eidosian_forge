from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
def _validate_rrsig(rrset, rrsig, keys, origin=None, now=None):
    """Validate an RRset against a single signature rdata

    The owner name of *rrsig* is assumed to be the same as the owner name
    of *rrset*.

    *rrset* is the RRset to validate.  It can be a ``dns.rrset.RRset`` or
    a ``(dns.name.Name, dns.rdataset.Rdataset)`` tuple.

    *rrsig* is a ``dns.rdata.Rdata``, the signature to validate.

    *keys* is the key dictionary, used to find the DNSKEY associated with
    a given name.  The dictionary is keyed by a ``dns.name.Name``, and has
    ``dns.node.Node`` or ``dns.rdataset.Rdataset`` values.

    *origin* is a ``dns.name.Name``, the origin to use for relative names.

    *now* is an ``int``, the time to use when validating the signatures,
    in seconds since the UNIX epoch.  The default is the current time.
    """
    if isinstance(origin, string_types):
        origin = dns.name.from_text(origin, dns.name.root)
    candidate_keys = _find_candidate_keys(keys, rrsig)
    if candidate_keys is None:
        raise ValidationFailure('unknown key')
    for candidate_key in candidate_keys:
        if isinstance(rrset, tuple):
            rrname = rrset[0]
            rdataset = rrset[1]
        else:
            rrname = rrset.name
            rdataset = rrset
        if now is None:
            now = time.time()
        if rrsig.expiration < now:
            raise ValidationFailure('expired')
        if rrsig.inception > now:
            raise ValidationFailure('not yet valid')
        hash = _make_hash(rrsig.algorithm)
        if _is_rsa(rrsig.algorithm):
            keyptr = candidate_key.key
            bytes_, = struct.unpack('!B', keyptr[0:1])
            keyptr = keyptr[1:]
            if bytes_ == 0:
                bytes_, = struct.unpack('!H', keyptr[0:2])
                keyptr = keyptr[2:]
            rsa_e = keyptr[0:bytes_]
            rsa_n = keyptr[bytes_:]
            try:
                pubkey = CryptoRSA.construct((number.bytes_to_long(rsa_n), number.bytes_to_long(rsa_e)))
            except ValueError:
                raise ValidationFailure('invalid public key')
            sig = rrsig.signature
        elif _is_dsa(rrsig.algorithm):
            keyptr = candidate_key.key
            t, = struct.unpack('!B', keyptr[0:1])
            keyptr = keyptr[1:]
            octets = 64 + t * 8
            dsa_q = keyptr[0:20]
            keyptr = keyptr[20:]
            dsa_p = keyptr[0:octets]
            keyptr = keyptr[octets:]
            dsa_g = keyptr[0:octets]
            keyptr = keyptr[octets:]
            dsa_y = keyptr[0:octets]
            pubkey = CryptoDSA.construct((number.bytes_to_long(dsa_y), number.bytes_to_long(dsa_g), number.bytes_to_long(dsa_p), number.bytes_to_long(dsa_q)))
            sig = rrsig.signature[1:]
        elif _is_ecdsa(rrsig.algorithm):
            keyptr = candidate_key.key
            if rrsig.algorithm == ECDSAP256SHA256:
                curve = ecdsa.curves.NIST256p
                key_len = 32
            elif rrsig.algorithm == ECDSAP384SHA384:
                curve = ecdsa.curves.NIST384p
                key_len = 48
            x = number.bytes_to_long(keyptr[0:key_len])
            y = number.bytes_to_long(keyptr[key_len:key_len * 2])
            if not ecdsa.ecdsa.point_is_valid(curve.generator, x, y):
                raise ValidationFailure('invalid ECDSA key')
            point = ecdsa.ellipticcurve.Point(curve.curve, x, y, curve.order)
            verifying_key = ecdsa.keys.VerifyingKey.from_public_point(point, curve)
            pubkey = ECKeyWrapper(verifying_key, key_len)
            r = rrsig.signature[:key_len]
            s = rrsig.signature[key_len:]
            sig = ecdsa.ecdsa.Signature(number.bytes_to_long(r), number.bytes_to_long(s))
        else:
            raise ValidationFailure('unknown algorithm %u' % rrsig.algorithm)
        hash.update(_to_rdata(rrsig, origin)[:18])
        hash.update(rrsig.signer.to_digestable(origin))
        if rrsig.labels < len(rrname) - 1:
            suffix = rrname.split(rrsig.labels + 1)[1]
            rrname = dns.name.from_text('*', suffix)
        rrnamebuf = rrname.to_digestable(origin)
        rrfixed = struct.pack('!HHI', rdataset.rdtype, rdataset.rdclass, rrsig.original_ttl)
        rrlist = sorted(rdataset)
        for rr in rrlist:
            hash.update(rrnamebuf)
            hash.update(rrfixed)
            rrdata = rr.to_digestable(origin)
            rrlen = struct.pack('!H', len(rrdata))
            hash.update(rrlen)
            hash.update(rrdata)
        try:
            if _is_rsa(rrsig.algorithm):
                verifier = pkcs1_15.new(pubkey)
                verifier.verify(hash, sig)
            elif _is_dsa(rrsig.algorithm):
                verifier = DSS.new(pubkey, 'fips-186-3')
                verifier.verify(hash, sig)
            elif _is_ecdsa(rrsig.algorithm):
                digest = hash.digest()
                if not pubkey.verify(digest, sig):
                    raise ValueError
            else:
                raise ValidationFailure('unknown algorithm %u' % rrsig.algorithm)
            return
        except ValueError:
            continue
    raise ValidationFailure('verify failure')