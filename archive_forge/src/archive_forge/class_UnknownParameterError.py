from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownParameterError(ValidationError):
    """
    Unknown top level parameter.

    :ivar name: The name of the unknown parameter.
    :ivar operation: The name of the operation.
    :ivar choices: The valid choices the parameter name can be.
    """
    fmt = "Unknown parameter '{name}' for operation {operation}.  Must be one of: {choices}"