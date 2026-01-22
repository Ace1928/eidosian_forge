import keystoneauth1.exceptions as kc_exception
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def _create_keypair(self):
    if self.data().get('credential_id'):
        return
    user_id = self._get_user_id()
    kp = self.keystone().create_stack_domain_user_keypair(user_id=user_id, project_id=self.stack.stack_user_project_id)
    if not kp:
        raise exception.Error(_('Error creating ec2 keypair for user %s') % user_id)
    else:
        try:
            credential_id = kp.id
        except AttributeError:
            credential_id = kp.access
        self.data_set('credential_id', credential_id, redact=True)
        self.data_set('access_key', kp.access, redact=True)
        self.data_set('secret_key', kp.secret, redact=True)
    return kp