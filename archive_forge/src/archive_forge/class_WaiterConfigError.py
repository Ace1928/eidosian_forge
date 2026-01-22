from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class WaiterConfigError(BotoCoreError):
    """Error when processing waiter configuration."""
    fmt = 'Error processing waiter config: {error_msg}'