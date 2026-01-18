from pprint import pformat
from six import iteritems
import re
@webhook.setter
def webhook(self, webhook):
    """
        Sets the webhook of this V1alpha1AuditSinkSpec.
        Webhook to send events required

        :param webhook: The webhook of this V1alpha1AuditSinkSpec.
        :type: V1alpha1Webhook
        """
    if webhook is None:
        raise ValueError('Invalid value for `webhook`, must not be `None`')
    self._webhook = webhook