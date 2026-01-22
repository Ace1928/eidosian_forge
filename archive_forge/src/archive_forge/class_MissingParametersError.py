from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class MissingParametersError(BotoCoreError):
    """
    One or more required parameters were not supplied.

    :ivar object: The object that has missing parameters.
        This can be an operation or a parameter (in the
        case of inner params).  The str() of this object
        will be used so it doesn't need to implement anything
        other than str().
    :ivar missing: The names of the missing parameters.
    """
    fmt = 'The following required parameters are missing for {object_name}: {missing}'