from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownServiceStyle(BotoCoreError):
    """
    Unknown style of service invocation.

    :ivar service_style: The style requested.
    """
    fmt = 'The service style ({service_style}) is not understood.'