from boto.exception import JSONResponseError
class ItemCollectionSizeLimitExceededException(JSONResponseError):
    pass