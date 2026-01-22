import collections
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.resources import signal_responder
def _get_ec2_signed_url(self, signal_type=signal_responder.WAITCONDITION):
    stored = self.data().get('ec2_signed_url')
    if stored is not None:
        return stored
    url = super(BaseWaitConditionHandle, self)._get_ec2_signed_url(signal_type)
    self.data_set('ec2_signed_url', url)
    return url