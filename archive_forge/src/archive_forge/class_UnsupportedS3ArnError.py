from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedS3ArnError(BotoCoreError):
    """Error when S3 ARN provided to Bucket parameter is not supported"""
    fmt = 'S3 ARN {arn} provided to "Bucket" parameter is invalid. Only ARNs for S3 access-points are supported.'