import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import credentials
class CredentialTests(utils.ClientTestCase, utils.CrudTests):

    def setUp(self):
        super(CredentialTests, self).setUp()
        self.key = 'credential'
        self.collection_key = 'credentials'
        self.model = credentials.Credential
        self.manager = self.client.credentials

    def new_ref(self, **kwargs):
        kwargs = super(CredentialTests, self).new_ref(**kwargs)
        kwargs.setdefault('blob', uuid.uuid4().hex)
        kwargs.setdefault('project_id', uuid.uuid4().hex)
        kwargs.setdefault('type', uuid.uuid4().hex)
        kwargs.setdefault('user_id', uuid.uuid4().hex)
        return kwargs