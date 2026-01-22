from boto.exception import JSONResponseError
class InsufficientS3BucketPolicy(JSONResponseError):
    pass