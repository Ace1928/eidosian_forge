import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
class MessagePartGatherSerializer:

    def dumps(self, obj):
        """
        The parameter is an already serialized list of Message objects. No need
        to serialize it again, only join the list together and encode it.
        """
        return ('[' + ','.join(obj) + ']').encode('latin-1')