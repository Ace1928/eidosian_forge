from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class EndpointConnectionError(ConnectionError):
    fmt = 'Could not connect to the endpoint URL: "{endpoint_url}"'