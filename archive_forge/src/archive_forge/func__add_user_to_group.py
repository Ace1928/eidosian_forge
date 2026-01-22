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
def _add_user_to_group(self, user_id, group_id):
    msg = _DEPRECATION_MSG % 'add_user_to_group'
    versionutils.report_deprecated_feature(LOG, msg)
    user_ref = self._get_user(user_id)
    user_dn = user_ref['dn']
    self.group.add_user(user_dn, group_id, user_id)