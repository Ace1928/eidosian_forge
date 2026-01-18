from .. import errors, mail_client, osutils, tests, urlutils
def test_to_required(self):
    claws = mail_client.Claws(None)
    self.assertRaises(mail_client.NoMailAddressSpecified, claws._get_compose_commandline, None, None, 'file%')