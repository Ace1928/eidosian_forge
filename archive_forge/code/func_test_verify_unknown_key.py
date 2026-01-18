import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
def test_verify_unknown_key(self):
    self.requireFeature(features.gpg)
    self.import_keys()
    content = b'-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nasdf\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJOORKwAAoJENf6AkFdUeVvJDYH/1Cz+AJn1Jvy5n64o+0fZ5Ow\nY7UQb4QQTIOV7jI7n4hv/yBzuHrtImFzYvQl/o2Ezzi8B8L5gZtQy+xCUF+Q8iWs\ngytZ5JUtSze7hDZo1NUl4etjoRGYqRfrUcvE2LkVH2dFbDGyyQfVmoeSHa5akuuP\nQZmyg2F983rACVIpGvsqTH6RcBdvE9vx68lugeKQA8ArDn39/74FBFipFzrXSPij\neKFpl+yZmIb3g6HkPIC8o4j/tMvc37xF1OG5sBu8FT0+FC+VgY7vAblneDftAbyP\nsIODx4WcfJtjLG/qkRYqJ4gDHo0eMpTJSk2CWebajdm4b+JBrM1F9mgKuZFLruE=\n=RNR5\n-----END PGP SIGNATURE-----\n'
    my_gpg = gpg.GPGStrategy(FakeConfig())
    self.assertEqual((gpg.SIGNATURE_KEY_MISSING, '5D51E56F', None), my_gpg.verify(content))