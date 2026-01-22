import tempfile
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional import base
class KeypairTests(KeypairBase):
    """Functional tests for compute keypairs."""
    PUBLIC_KEY = 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDWNGczJxNaFUrJJVhta4dWsZY6bU5HUMPbyfSMu713ca3mYtG848W4dfDCB98KmSQx2Bl0D6Q2nrOszOXEQWAXNdfMadnWc4mNwhZcPBVohIFoC1KZJC8kcBTvFZcoz3mdIijxJtywZNpGNh34VRJlZeHyYjg8/DesHzdoBVd5c/4R36emQSIV9ukY6PHeZ3scAH4B3K9PxItJBwiFtouSRphQG0bJgOv/gjAjMElAvg5oku98cb4QiHZ8T8WY68id804raHR6pJxpVVJN4TYJmlUs+NOVM+pPKbKJttqrIBTkawGK9pLHNfn7z6v1syvUo/4enc1l0Q/Qn2kWiz67 fake@openstack'

    def setUp(self):
        """Create keypair with randomized name for tests."""
        super(KeypairTests, self).setUp()
        self.KPName = data_utils.rand_name('TestKeyPair')
        self.keypair = self.keypair_create(self.KPName)

    def test_keypair_create_duplicate(self):
        """Try to create duplicate name keypair.

        Test steps:
        1) Create keypair in setUp
        2) Try to create duplicate keypair with the same name
        """
        self.assertRaises(exceptions.CommandFailed, self.openstack, 'keypair create ' + self.KPName)

    def test_keypair_create_noname(self):
        """Try to create keypair without name.

        Test steps:
        1) Try to create keypair without a name
        """
        self.assertRaises(exceptions.CommandFailed, self.openstack, 'keypair create')

    def test_keypair_create_public_key(self):
        """Test for create keypair with --public-key option.

        Test steps:
        1) Create keypair with given public key
        2) Delete keypair
        """
        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write(self.PUBLIC_KEY)
            f.flush()
            raw_output = self.openstack('keypair create --public-key %s tmpkey' % f.name)
            self.addCleanup(self.openstack, 'keypair delete tmpkey')
            self.assertIn('tmpkey', raw_output)

    def test_keypair_create_private_key(self):
        """Test for create keypair with --private-key option.

        Test steps:
        1) Create keypair with private key file
        2) Delete keypair
        """
        with tempfile.NamedTemporaryFile(mode='w+') as f:
            cmd_output = self.openstack('keypair create --private-key %s tmpkey' % f.name, parse_output=True)
            self.addCleanup(self.openstack, 'keypair delete tmpkey')
            self.assertEqual('tmpkey', cmd_output.get('name'))
            self.assertIsNotNone(cmd_output.get('user_id'))
            self.assertIsNotNone(cmd_output.get('fingerprint'))
            pk_content = f.read()
            self.assertInOutput('-----BEGIN OPENSSH PRIVATE KEY-----', pk_content)
            self.assertRegex(pk_content, '[0-9A-Za-z+/]+[=]{0,3}\n')
            self.assertInOutput('-----END OPENSSH PRIVATE KEY-----', pk_content)

    def test_keypair_create(self):
        """Test keypair create command.

        Test steps:
        1) Create keypair in setUp
        2) Check Ed25519 private key in output
        3) Check for new keypair in keypairs list
        """
        NewName = data_utils.rand_name('TestKeyPairCreated')
        raw_output = self.openstack('keypair create ' + NewName)
        self.addCleanup(self.openstack, 'keypair delete ' + NewName)
        self.assertInOutput('-----BEGIN OPENSSH PRIVATE KEY-----', raw_output)
        self.assertRegex(raw_output, '[0-9A-Za-z+/]+[=]{0,3}\n')
        self.assertInOutput('-----END OPENSSH PRIVATE KEY-----', raw_output)
        self.assertIn(NewName, self.keypair_list())

    def test_keypair_delete_not_existing(self):
        """Try to delete keypair with not existing name.

        Test steps:
        1) Create keypair in setUp
        2) Try to delete not existing keypair
        """
        self.assertRaises(exceptions.CommandFailed, self.openstack, 'keypair delete not_existing')

    def test_keypair_delete(self):
        """Test keypair delete command.

        Test steps:
        1) Create keypair in setUp
        2) Delete keypair
        3) Check that keypair not in keypairs list
        """
        self.openstack('keypair delete ' + self.KPName)
        self.assertNotIn(self.KPName, self.keypair_list())

    def test_keypair_list(self):
        """Test keypair list command.

        Test steps:
        1) Create keypair in setUp
        2) List keypairs
        3) Check output table structure
        4) Check keypair name in output
        """
        HEADERS = ['Name', 'Fingerprint']
        raw_output = self.openstack('keypair list')
        items = self.parse_listing(raw_output)
        self.assert_table_structure(items, HEADERS)
        self.assertIn(self.KPName, raw_output)

    def test_keypair_show(self):
        """Test keypair show command.

        Test steps:
        1) Create keypair in setUp
        2) Show keypair
        3) Check output table structure
        4) Check keypair name in output
        """
        HEADERS = ['Field', 'Value']
        raw_output = self.openstack('keypair show ' + self.KPName)
        items = self.parse_listing(raw_output)
        self.assert_table_structure(items, HEADERS)
        self.assertInOutput(self.KPName, raw_output)