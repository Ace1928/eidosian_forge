from castellan.common.credentials import password
from castellan.tests import base
def _create_password_credential(self):
    return password.Password(self.username, self.password)