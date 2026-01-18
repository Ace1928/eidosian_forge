import warnings
from passlib.tests.utils import TestCase
from passlib.utils.compat import u
def test_nthash(self):
    warnings.filterwarnings('ignore', 'nthash\\.raw_nthash\\(\\) is deprecated')
    from passlib.win32 import raw_nthash
    for secret, hash in [('OLDPASSWORD', u('6677b2c394311355b54f25eec5bfacf5')), ('NEWPASSWORD', u('256781a62031289d3c2c98c14f1efc8c'))]:
        result = raw_nthash(secret, hex=True)
        self.assertEqual(result, hash)