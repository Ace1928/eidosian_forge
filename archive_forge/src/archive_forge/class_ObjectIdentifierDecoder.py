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
class ObjectIdentifierDecoder(AbstractSimpleDecoder):
    protoComponent = univ.ObjectIdentifier(())

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')
        head, tail = (substrate[:length], substrate[length:])
        if not head:
            raise error.PyAsn1Error('Empty substrate')
        head = octs2ints(head)
        oid = ()
        index = 0
        substrateLen = len(head)
        while index < substrateLen:
            subId = head[index]
            index += 1
            if subId < 128:
                oid += (subId,)
            elif subId > 128:
                nextSubId = subId
                subId = 0
                while nextSubId >= 128:
                    subId = (subId << 7) + (nextSubId & 127)
                    if index >= substrateLen:
                        raise error.SubstrateUnderrunError('Short substrate for sub-OID past %s' % (oid,))
                    nextSubId = head[index]
                    index += 1
                oid += ((subId << 7) + nextSubId,)
            elif subId == 128:
                raise error.PyAsn1Error('Invalid octet 0x80 in OID encoding')
        if 0 <= oid[0] <= 39:
            oid = (0,) + oid
        elif 40 <= oid[0] <= 79:
            oid = (1, oid[0] - 40) + oid[1:]
        elif oid[0] >= 80:
            oid = (2, oid[0] - 80) + oid[1:]
        else:
            raise error.PyAsn1Error('Malformed first OID octet: %s' % head[0])
        return (self._createComponent(asn1Spec, tagSet, oid, **options), tail)