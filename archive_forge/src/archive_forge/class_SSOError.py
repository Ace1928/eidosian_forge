from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class SSOError(BotoCoreError):
    fmt = 'An unspecified error happened when resolving AWS credentials or an access token from SSO.'