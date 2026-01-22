from boto.exception import BotoServerError
class InvalidS3BucketNameException(BotoServerError):
    """
    Raised when an invalid S3 bucket name is passed to Cloudtrail.
    """
    pass