import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
def test_verify_expired_but_valid(self):
    self.requireFeature(features.gpg)
    self.import_keys()
    content = b'-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: test@example.com-20110801100657-f1dr1nompeex723z\nsha1: 59ab434be4c2d5d646dee84f514aa09e1b72feeb\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.10 (GNU/Linux)\n\niJwEAQECAAYFAk42esUACgkQHOJve0+NFRPc5wP7BoZkzBU8JaHMLv/LmqLr0sUz\nzuE51ofZZ19L7KVtQWsOi4jFy0fi4A5TFwO8u9SOfoREGvkw292Uty9subSouK5/\nmFmDOYPQ+O83zWgYZsBmMJWYDZ+X9I6XXZSbPtV/7XyTjaxtl5uRnDVJjg+AzKvD\ndTp8VatVVrwuvzOPDVc=\n=uHen\n-----END PGP SIGNATURE-----\n'
    my_gpg = gpg.GPGStrategy(FakeConfig())
    self.assertEqual((gpg.SIGNATURE_EXPIRED, '4F8D1513', None), my_gpg.verify(content))