from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import stack_user
def _secret_accesskey(self):
    """Return the user's access key.

        Fetching it from keystone if necessary.
        """
    if self._secret is None:
        if not self.resource_id:
            LOG.info('could not get secret for %(username)s Error:%(msg)s', {'username': self.properties[self.USER_NAME], 'msg': 'resource_id not yet set'})
        else:
            self._secret = self.data().get('secret_key')
            if self._secret is None:
                try:
                    user_id = self._get_user().resource_id
                    kp = self.keystone().get_ec2_keypair(user_id=user_id, access=self.resource_id)
                    self._secret = kp.secret
                    self.data_set('secret_key', kp.secret, redact=True)
                    self.data_set('credential_id', kp.id, redact=True)
                except Exception as ex:
                    LOG.info('could not get secret for %(username)s Error:%(msg)s', {'username': self.properties[self.USER_NAME], 'msg': ex})
    return self._secret or '000-000-000'