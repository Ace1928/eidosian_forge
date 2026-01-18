import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
def test_verify_unacceptable_key(self):
    self.requireFeature(features.gpg)
    self.import_keys()
    content = b'-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nbazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niQEcBAEBAgAGBQJN+ekFAAoJEIdoGx7jCA5FGtEH/i+XxJRvqU6wdBtLVrGBMAGk\nFZ5VP+KyXYtymSbgSstj/vM12NeMIeFs3xGnNnYuX1MIcY6We5TKtCH0epY6ym5+\n6g2Q2QpQ5/sT2d0mWzR0K4uVngmxVQaXTdk5PdZ40O7ULeDLW6CxzxMHyUL1rsIx\n7UBUTBh1O/1n3ZfD99hUkm3hVcnsN90uTKH59zV9NWwArU0cug60+5eDKJhSJDbG\nrIwlqbFAjDZ7L/48e+IaYIJwBZFzMBpJKdCxzALLtauMf+KK8hGiL2hrRbWm7ty6\nNgxfkMYOB4rDPdSstT35N+5uBG3n/UzjxHssi0svMfVETYYX40y57dm2eZQXFp8=\n=iwsn\n-----END PGP SIGNATURE-----\n'
    plain = b'bazaar-ng testament short form 1\nrevision-id: amy@example.com-20110527185938-hluafawphszb8dl1\nsha1: 6411f9bdf6571200357140c9ce7c0f50106ac9a4\n'
    my_gpg = gpg.GPGStrategy(FakeConfig())
    my_gpg.set_acceptable_keys('foo@example.com')
    self.assertEqual((gpg.SIGNATURE_KEY_MISSING, 'E3080E45', plain), my_gpg.verify(content))