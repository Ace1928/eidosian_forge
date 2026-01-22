from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class TokenRetrievalError(BotoCoreError):
    """
    Error attempting to retrieve a token from a remote source.

    :ivar provider: The name of the token provider.
    :ivar error_msg: The msg explaining why the token could not be retrieved.

    """
    fmt = 'Error when retrieving token from {provider}: {error_msg}'