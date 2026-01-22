from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownCredentialError(BotoCoreError):
    """Tried to insert before/after an unregistered credential type."""
    fmt = 'Credential named {name} not found.'