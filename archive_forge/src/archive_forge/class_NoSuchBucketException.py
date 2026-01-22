from boto.exception import BotoServerError
class NoSuchBucketException(BotoServerError):
    pass