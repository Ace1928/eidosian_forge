import sys
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.codec.der import encoder as der_encoder
from pyasn1_modules import pem
from pyasn1_modules import rfc2459
class DSAPrivateKeyTestCase(unittest.TestCase):
    pem_text = 'MIIBugIBAAKBgQCN91+Cma8UPw09gjwP9WOJCdpv3mv3/qFqzgiODGZx0Q002iTl\n1dq36m5TsWYFEcMCEyC3tFuoQ0mGq5zUUOmJvHCIPufs0g8Av0fhY77uFqneHHUi\nVQMCPCHX9vTCWskmDE21LJppU27bR4H2q+ysE30d6u3+84qrItsn4bjpcQIVAPR5\nQrmooOXDn7fHJzshmxImGC4VAoGAXxKyEnlvzq93d4V6KLWX3H5Jk2JP771Ss1bT\n6D/mSbLlvjjo7qsj6diul1axu6Wny31oPertzA2FeGEzkqvjSNmSxyYYMDB3kEcx\nahntt37I1FgSlgdZHuhdtl1h1DBKXqCCneOZuNj+kW5ib14u5HDfFIbec2HJbvVs\nlJ/k83kCgYB4TD8vgHetXHxqsiZDoy5wOnQ3mmFAfl8ZdQsIfov6kEgArwPYUOVB\nJsX84f+MFjIOKXUV8dHZ8VRrGCLAbXcxKqLNWKlKHUnEsvt63pkaTy/RKHyQS+pn\nwontdTt9EtbF+CqIWnm2wpn3O+SbdtawzPOL1CcGB0jYABwbeQ81RwIUFKdyRYaa\nINow2I3/ks+0MxDabTY=\n'

    def setUp(self):
        self.asn1Spec = rfc2459.DSAPrivateKey()

    def testDerCodec(self):
        substrate = pem.readBase64fromText(self.pem_text)
        asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec)
        assert not rest
        assert asn1Object.prettyPrint()
        assert der_encoder.encode(asn1Object) == substrate

    def testDerCodecDecodeOpenTypes(self):
        substrate = pem.readBase64fromText(self.pem_text)
        asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec, decodeOpenTypes=True)
        assert not rest
        assert asn1Object.prettyPrint()
        assert der_encoder.encode(asn1Object) == substrate