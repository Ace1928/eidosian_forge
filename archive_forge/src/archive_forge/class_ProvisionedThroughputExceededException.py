from boto.exception import JSONResponseError
class ProvisionedThroughputExceededException(JSONResponseError):
    pass