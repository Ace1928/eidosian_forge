from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class RealDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Real()

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')
        head, tail = (substrate[:length], substrate[length:])
        if not head:
            return (self._createComponent(asn1Spec, tagSet, 0.0, **options), tail)
        fo = oct2int(head[0])
        head = head[1:]
        if fo & 128:
            if not head:
                raise error.PyAsn1Error('Incomplete floating-point value')
            n = (fo & 3) + 1
            if n == 4:
                n = oct2int(head[0])
                head = head[1:]
            eo, head = (head[:n], head[n:])
            if not eo or not head:
                raise error.PyAsn1Error('Real exponent screwed')
            e = oct2int(eo[0]) & 128 and -1 or 0
            while eo:
                e <<= 8
                e |= oct2int(eo[0])
                eo = eo[1:]
            b = fo >> 4 & 3
            if b > 2:
                raise error.PyAsn1Error('Illegal Real base')
            if b == 1:
                e *= 3
            elif b == 2:
                e *= 4
            p = 0
            while head:
                p <<= 8
                p |= oct2int(head[0])
                head = head[1:]
            if fo & 64:
                p = -p
            sf = fo >> 2 & 3
            p *= 2 ** sf
            value = (p, 2, e)
        elif fo & 64:
            value = fo & 1 and '-inf' or 'inf'
        elif fo & 192 == 0:
            if not head:
                raise error.PyAsn1Error('Incomplete floating-point value')
            try:
                if fo & 3 == 1:
                    value = (int(head), 10, 0)
                elif fo & 3 == 2:
                    value = float(head)
                elif fo & 3 == 3:
                    value = float(head)
                else:
                    raise error.SubstrateUnderrunError('Unknown NR (tag %s)' % fo)
            except ValueError:
                raise error.SubstrateUnderrunError('Bad character Real syntax')
        else:
            raise error.SubstrateUnderrunError('Unknown encoding (tag %s)' % fo)
        return (self._createComponent(asn1Spec, tagSet, value, **options), tail)