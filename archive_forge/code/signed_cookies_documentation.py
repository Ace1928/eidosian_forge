from django.contrib.sessions.backends.base import SessionBase
from django.core import signing

        Instead of generating a random string, generate a secure url-safe
        base64-encoded string of data as our session key.
        