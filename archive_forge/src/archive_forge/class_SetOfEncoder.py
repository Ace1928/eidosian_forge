from pyasn1 import error
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import str2octs, null
from pyasn1.type import univ
from pyasn1.type import useful
class SetOfEncoder(encoder.SequenceOfEncoder):

    def encodeValue(self, value, asn1Spec, encodeFun, **options):
        if asn1Spec is None:
            value.verifySizeSpec()
        else:
            asn1Spec = asn1Spec.componentType
        components = [encodeFun(x, asn1Spec, **options) for x in value]
        if len(components) > 1:
            zero = str2octs('\x00')
            maxLen = max(map(len, components))
            paddedComponents = [(x.ljust(maxLen, zero), x) for x in components]
            paddedComponents.sort(key=lambda x: x[0])
            components = [x[1] for x in paddedComponents]
        substrate = null.join(components)
        return (substrate, True, True)