from boto.exception import JSONResponseError
class UnauthorizedOperation(JSONResponseError):
    pass