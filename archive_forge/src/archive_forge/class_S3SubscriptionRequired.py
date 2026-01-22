import sys
from boto.compat import json
from boto.exception import BotoServerError
class S3SubscriptionRequired(SimpleException):
    pass