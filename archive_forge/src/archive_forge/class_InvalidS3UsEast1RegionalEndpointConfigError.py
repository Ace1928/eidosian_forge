from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidS3UsEast1RegionalEndpointConfigError(BotoCoreError):
    """Error for invalid s3 us-east-1 regional endpoints configuration"""
    fmt = 'S3 us-east-1 regional endpoint option {s3_us_east_1_regional_endpoint_config} is invalid. Valid options are: "legacy", "regional"'