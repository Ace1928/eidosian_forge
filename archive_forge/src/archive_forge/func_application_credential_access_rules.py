import datetime
import uuid
from keystoneauth1 import _utils
from keystoneauth1.fixture import exception
@application_credential_access_rules.setter
def application_credential_access_rules(self, value):
    application_credential = self.root.setdefault('application_credential', {})
    application_credential.setdefault('access_rules', value)