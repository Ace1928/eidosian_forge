from boto.exception import JSONResponseError
class ReservedNodeNotFoundFault(JSONResponseError):
    pass