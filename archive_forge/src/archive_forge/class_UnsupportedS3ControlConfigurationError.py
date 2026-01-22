from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedS3ControlConfigurationError(BotoCoreError):
    """Error when an unsupported configuration is used with S3 Control"""
    fmt = 'Unsupported configuration when using S3 Control: {msg}'