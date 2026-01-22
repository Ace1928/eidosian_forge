from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidDefaultsMode(BotoCoreError):
    fmt = 'Client configured with invalid defaults mode: {mode}. Valid defaults modes include: {valid_modes}.'