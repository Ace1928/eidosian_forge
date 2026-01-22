import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
def _update_cookie(self, encoded_data, response):
    """
        Either set the cookie with the encoded data if there is any data to
        store, or delete the cookie.
        """
    if encoded_data:
        response.set_cookie(self.cookie_name, encoded_data, domain=settings.SESSION_COOKIE_DOMAIN, secure=settings.SESSION_COOKIE_SECURE or None, httponly=settings.SESSION_COOKIE_HTTPONLY or None, samesite=settings.SESSION_COOKIE_SAMESITE)
    else:
        response.delete_cookie(self.cookie_name, domain=settings.SESSION_COOKIE_DOMAIN, samesite=settings.SESSION_COOKIE_SAMESITE)