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
class NullDecoder(AbstractSimpleDecoder):
    protoComponent = univ.Null('')

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatSimple:
            raise error.PyAsn1Error('Simple tag format expected')
        head, tail = (substrate[:length], substrate[length:])
        component = self._createComponent(asn1Spec, tagSet, '', **options)
        if head:
            raise error.PyAsn1Error('Unexpected %d-octet substrate for Null' % length)
        return (component, tail)