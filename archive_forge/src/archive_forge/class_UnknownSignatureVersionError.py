from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownSignatureVersionError(BotoCoreError):
    """
    Requested Signature Version is not known.

    :ivar signature_version: The name of the requested signature version.
    """
    fmt = 'Unknown Signature Version: {signature_version}.'