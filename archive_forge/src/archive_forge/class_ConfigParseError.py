from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class ConfigParseError(BotoCoreError):
    """
    The configuration file could not be parsed.

    :ivar path: The path to the configuration file.
    """
    fmt = 'Unable to parse config file: {path}'