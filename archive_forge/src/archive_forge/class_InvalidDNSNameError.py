from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidDNSNameError(BotoCoreError):
    """Error when virtual host path is forced on a non-DNS compatible bucket"""
    fmt = "Bucket named {bucket_name} is not DNS compatible. Virtual hosted-style addressing cannot be used. The addressing style can be configured by removing the addressing_style value or setting that value to 'path' or 'auto' in the AWS Config file or in the botocore.client.Config object."