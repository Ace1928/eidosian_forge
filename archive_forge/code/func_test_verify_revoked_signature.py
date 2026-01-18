import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
def test_verify_revoked_signature(self):
    self.requireFeature(features.gpg)
    self.import_keys()
    content = b'-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\nasdf\n-----BEGIN PGP SIGNATURE-----\nVersion: GnuPG v1.4.11 (GNU/Linux)\n\niJwEAQECAAYFAk45V18ACgkQjs6dvEpb0cSIZQP/eOGTXGPlrNwvDkcX2d8O///I\necB4sUIUEpv1XAk1MkNu58lsjjK72lRaLusEGqd7HwrFmpxVeVs0oWLg23PNPCFs\nyJBID9ma+VxFVPtkEFnrc1R72sBJLfBcTxMkwVTC8eeznjdtn+cg+aLkxbPdrGnr\nJFA6kUIJU2w9LU/b88Y=\n=UuRX\n-----END PGP SIGNATURE-----\n'
    plain = b'asdf\n'
    my_gpg = gpg.GPGStrategy(FakeConfig())
    my_gpg.set_acceptable_keys('test@example.com')
    self.assertEqual((gpg.SIGNATURE_NOT_VALID, None, None), my_gpg.verify(content))