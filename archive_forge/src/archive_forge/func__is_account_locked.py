import datetime
from oslo_db import api as oslo_db_api
import sqlalchemy
from keystone.common import driver_hints
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
def _is_account_locked(self, user_id, user_ref):
    """Check if the user account is locked.

        Checks if the user account is locked based on the number of failed
        authentication attempts.

        :param user_id: The user ID
        :param user_ref: Reference to the user object
        :returns Boolean: True if the account is locked; False otherwise

        """
    ignore_option = user_ref.get_resource_option(options.IGNORE_LOCKOUT_ATTEMPT_OPT.option_id)
    if ignore_option and ignore_option.option_value is True:
        return False
    attempts = user_ref.local_user.failed_auth_count or 0
    max_attempts = CONF.security_compliance.lockout_failure_attempts
    lockout_duration = CONF.security_compliance.lockout_duration
    if max_attempts and attempts >= max_attempts:
        if not lockout_duration:
            return True
        else:
            delta = datetime.timedelta(seconds=lockout_duration)
            last_failure = user_ref.local_user.failed_auth_at
            if last_failure + delta > datetime.datetime.utcnow():
                return True
            else:
                self._reset_failed_auth(user_id)
    return False