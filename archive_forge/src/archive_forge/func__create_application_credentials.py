from openstack import exceptions
from openstack.tests.functional import base
def _create_application_credentials(self):
    app_creds = self.conn.identity.create_application_credential(user=self.user_id, name='app_cred')
    self.addCleanup(self.conn.identity.delete_application_credential, self.user_id, app_creds['id'])
    return app_creds