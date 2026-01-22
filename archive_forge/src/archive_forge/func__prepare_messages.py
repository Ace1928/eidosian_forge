from django.conf import settings
from django.contrib.messages import constants, utils
from django.utils.functional import SimpleLazyObject
def _prepare_messages(self, messages):
    """
        Prepare a list of messages for storage.
        """
    for message in messages:
        message._prepare()