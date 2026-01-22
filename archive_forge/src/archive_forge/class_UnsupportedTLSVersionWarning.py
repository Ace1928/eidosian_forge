from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedTLSVersionWarning(Warning):
    """Warn when an openssl version that uses TLS 1.2 is required"""
    pass