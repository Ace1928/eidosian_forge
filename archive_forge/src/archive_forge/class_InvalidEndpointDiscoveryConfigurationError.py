from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidEndpointDiscoveryConfigurationError(BotoCoreError):
    """Error when invalid value supplied for endpoint_discovery_enabled"""
    fmt = 'Unsupported configuration value for endpoint_discovery_enabled. Expected one of ("true", "false", "auto") but got {config_value}.'