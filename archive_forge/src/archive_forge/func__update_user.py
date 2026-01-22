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
def _update_user(self, user_id, user):
    msg = _DEPRECATION_MSG % 'update_user'
    versionutils.report_deprecated_feature(LOG, msg)
    old_obj = self.user.get(user_id)
    if 'name' in user and old_obj.get('name') != user['name']:
        raise exception.Conflict(_('Cannot change user name'))
    if self.user.enabled_mask:
        self.user.mask_enabled_attribute(user)
    elif self.user.enabled_invert and (not self.user.enabled_emulation):
        user['enabled'] = not user['enabled']
        old_obj['enabled'] = not old_obj['enabled']
    self.user.update(user_id, user, old_obj)
    return self.user.get_filtered(user_id)