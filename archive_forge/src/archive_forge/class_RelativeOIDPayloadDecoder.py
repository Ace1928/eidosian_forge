import io
import os
import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.codec.streaming import asSeekableStream
from pyasn1.codec.streaming import isEndOfStream
from pyasn1.codec.streaming import peekIntoStream
from pyasn1.codec.streaming import readFromStream
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.error import PyAsn1Error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class RelativeOIDPayloadDecoder(AbstractSimplePayloadDecoder):
    protoComponent = univ.RelativeOID(())

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')
        for chunk in readFromStream(substrate, length, options):
            if isinstance(chunk, SubstrateUnderrunError):
                yield chunk
        if not chunk:
            raise error.PyAsn1Error('Empty substrate')
        chunk = octs2ints(chunk)
        reloid = ()
        index = 0
        substrateLen = len(chunk)
        while index < substrateLen:
            subId = chunk[index]
            index += 1
            if subId < 128:
                reloid += (subId,)
            elif subId > 128:
                nextSubId = subId
                subId = 0
                while nextSubId >= 128:
                    subId = (subId << 7) + (nextSubId & 127)
                    if index >= substrateLen:
                        raise error.SubstrateUnderrunError('Short substrate for sub-OID past %s' % (reloid,))
                    nextSubId = chunk[index]
                    index += 1
                reloid += ((subId << 7) + nextSubId,)
            elif subId == 128:
                raise error.PyAsn1Error('Invalid octet 0x80 in RELATIVE-OID encoding')
        yield self._createComponent(asn1Spec, tagSet, reloid, **options)