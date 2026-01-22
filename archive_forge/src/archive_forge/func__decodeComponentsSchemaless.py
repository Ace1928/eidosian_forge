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
def _decodeComponentsSchemaless(self, substrate, tagSet=None, decodeFun=None, length=None, **options):
    asn1Object = None
    components = []
    componentTypes = set()
    original_position = substrate.tell()
    while length == -1 or substrate.tell() < original_position + length:
        for component in decodeFun(substrate, **options):
            if isinstance(component, SubstrateUnderrunError):
                yield component
        if length == -1 and component is eoo.endOfOctets:
            break
        components.append(component)
        componentTypes.add(component.tagSet)
        if len(componentTypes) > 1:
            protoComponent = self.protoRecordComponent
        else:
            protoComponent = self.protoSequenceComponent
        asn1Object = protoComponent.clone(tagSet=tag.TagSet(protoComponent.tagSet.baseTag, *tagSet.superTags))
    if LOG:
        LOG('guessed %r container type (pass `asn1Spec` to guide the decoder)' % asn1Object)
    for idx, component in enumerate(components):
        asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
    yield asn1Object