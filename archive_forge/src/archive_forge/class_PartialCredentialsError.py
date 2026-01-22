from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class PartialCredentialsError(BotoCoreError):
    """
    Only partial credentials were found.

    :ivar cred_var: The missing credential variable name.

    """
    fmt = 'Partial credentials found in {provider}, missing: {cred_var}'