from castellan.common.credentials import token
from castellan.tests import base
class TokenTestCase(base.TestCase):

    def _create_token_credential(self):
        return token.Token(self.token)

    def setUp(self):
        self.token = '8a4aa147d58141c39a7a22905b90ba4e'
        self.token_credential = self._create_token_credential()
        super(TokenTestCase, self).setUp()

    def test_get_token(self):
        self.assertEqual(self.token, self.token_credential.token)

    def test___eq__(self):
        self.assertTrue(self.token_credential == self.token_credential)
        self.assertTrue(self.token_credential is self.token_credential)
        self.assertFalse(self.token_credential is None)
        self.assertFalse(None == self.token_credential)
        other_token_credential = token.Token(self.token)
        self.assertTrue(self.token_credential == other_token_credential)
        self.assertFalse(self.token_credential is other_token_credential)

    def test___ne___none(self):
        self.assertTrue(self.token_credential is not None)
        self.assertTrue(None != self.token_credential)

    def test___ne___token(self):
        other_token = 'fe32af1fe47e4744a48254e60ae80012'
        other_token_credential = token.Token(other_token)
        self.assertTrue(self.token_credential != other_token_credential)