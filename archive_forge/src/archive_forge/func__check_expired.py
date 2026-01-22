import datetime
import sqlalchemy
from keystone.application_credential.backends import base
from keystone.common import password_hashing
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _check_expired(self, app_cred_ref):
    if app_cred_ref.get('expires_at'):
        return datetime.datetime.utcnow() >= app_cred_ref['expires_at']
    return False