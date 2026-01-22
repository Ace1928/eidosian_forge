from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
class TOTPUserInfo(BaseUserInfo):

    def __init__(self):
        super(TOTPUserInfo, self).__init__()
        self.passcode = None

    def _validate_and_normalize_auth_data(self, auth_payload):
        super(TOTPUserInfo, self)._validate_and_normalize_auth_data(auth_payload)
        user_info = auth_payload['user']
        self.passcode = user_info.get('passcode')