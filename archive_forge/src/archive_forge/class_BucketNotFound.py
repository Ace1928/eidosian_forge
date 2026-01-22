from boto.exception import JSONResponseError
class BucketNotFound(JSONResponseError):
    pass