from pyasn1.codec.cer import decoder
from pyasn1.type import univ
class OctetStringDecoder(decoder.OctetStringDecoder):
    supportConstructedForm = False