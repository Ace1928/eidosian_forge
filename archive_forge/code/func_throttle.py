from pprint import pformat
from six import iteritems
import re
@throttle.setter
def throttle(self, throttle):
    """
        Sets the throttle of this V1alpha1Webhook.
        Throttle holds the options for throttling the webhook

        :param throttle: The throttle of this V1alpha1Webhook.
        :type: V1alpha1WebhookThrottleConfig
        """
    self._throttle = throttle