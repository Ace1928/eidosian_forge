import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
def _get_password_expires_at(self, created_at):
    expires_days = CONF.security_compliance.password_expires_days
    if not self._password_expiry_exempt():
        if expires_days:
            expired_date = created_at + datetime.timedelta(days=expires_days)
            return expired_date.replace(microsecond=0)
    return None