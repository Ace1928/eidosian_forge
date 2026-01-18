from pprint import pformat
from six import iteritems
import re
@webhook_client_config.setter
def webhook_client_config(self, webhook_client_config):
    """
        Sets the webhook_client_config of this V1beta1CustomResourceConversion.
        `webhookClientConfig` is the instructions for how to call the webhook if
        strategy is `Webhook`. This field is alpha-level and is only honored by
        servers that enable the CustomResourceWebhookConversion feature.

        :param webhook_client_config: The webhook_client_config of this
        V1beta1CustomResourceConversion.
        :type: ApiextensionsV1beta1WebhookClientConfig
        """
    self._webhook_client_config = webhook_client_config