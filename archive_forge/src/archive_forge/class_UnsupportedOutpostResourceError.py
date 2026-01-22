from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedOutpostResourceError(BotoCoreError):
    """Error when S3 Outpost ARN provided to Bucket parameter is incomplete"""
    fmt = 'S3 Outpost ARN resource "{resource_name}" provided to "Bucket" parameter is invalid. Only ARNs for S3 Outpost arns with an access-point sub-resource are supported.'