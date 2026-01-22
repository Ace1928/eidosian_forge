from boto.exception import JSONResponseError
class InternalServiceError(JSONResponseError):
    pass