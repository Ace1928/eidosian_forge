from pyasn1 import error
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import str2octs, null
from pyasn1.type import univ
from pyasn1.type import useful
class TimeEncoderMixIn(object):
    zchar, = str2octs('Z')
    pluschar, = str2octs('+')
    minuschar, = str2octs('-')
    commachar, = str2octs(',')
    minLength = 12
    maxLength = 19

    def encodeValue(self, value, asn1Spec, encodeFun, **options):
        if asn1Spec is not None:
            value = asn1Spec.clone(value)
        octets = value.asOctets()
        if not self.minLength < len(octets) < self.maxLength:
            raise error.PyAsn1Error('Length constraint violated: %r' % value)
        if self.pluschar in octets or self.minuschar in octets:
            raise error.PyAsn1Error('Must be UTC time: %r' % octets)
        if octets[-1] != self.zchar:
            raise error.PyAsn1Error('Missing "Z" time zone specifier: %r' % octets)
        if self.commachar in octets:
            raise error.PyAsn1Error('Comma in fractions disallowed: %r' % value)
        options.update(maxChunkSize=1000)
        return encoder.OctetStringEncoder.encodeValue(self, value, asn1Spec, encodeFun, **options)