import uuid
import ldap.filter
from oslo_log import log
from oslo_log import versionutils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends.ldap import models
def _create_user(self, user_id, user):
    msg = _DEPRECATION_MSG % 'create_user'
    versionutils.report_deprecated_feature(LOG, msg)
    user_ref = self.user.create(user)
    return self.user.filter_attributes(user_ref)