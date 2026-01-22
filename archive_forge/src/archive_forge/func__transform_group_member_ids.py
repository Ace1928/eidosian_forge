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
def _transform_group_member_ids(self, group_member_list):
    for user_key in group_member_list:
        if self.conf.ldap.group_members_are_ids:
            user_id = user_key
        else:
            user_id = self.user._dn_to_id(user_key)
        yield user_id