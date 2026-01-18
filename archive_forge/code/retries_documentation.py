from functools import wraps
from .cloud import CloudRetry

    Allow for boto3 not being installed when using these utils by wrapping
    botocore.exceptions instead of assigning from it directly.
    