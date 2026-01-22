from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnauthorizedSSOTokenError(SSOError):
    fmt = 'The SSO session associated with this profile has expired or is otherwise invalid. To refresh this SSO session run aws sso login with the corresponding profile.'