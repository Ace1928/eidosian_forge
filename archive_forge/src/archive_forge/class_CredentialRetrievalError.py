from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class CredentialRetrievalError(BotoCoreError):
    """
    Error attempting to retrieve credentials from a remote source.

    :ivar provider: The name of the credential provider.
    :ivar error_msg: The msg explaining why credentials could not be
        retrieved.

    """
    fmt = 'Error when retrieving credentials from {provider}: {error_msg}'