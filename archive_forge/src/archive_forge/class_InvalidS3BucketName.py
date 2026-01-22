from boto.exception import JSONResponseError
class InvalidS3BucketName(JSONResponseError):
    pass