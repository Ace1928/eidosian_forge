from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidRetryModeError(InvalidRetryConfigurationError):
    """Error when invalid retry mode configuration is specified"""
    fmt = 'Invalid value provided to "mode": "{provided_retry_mode}" must be one of: {valid_modes}'